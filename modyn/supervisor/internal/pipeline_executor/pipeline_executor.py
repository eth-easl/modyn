import json
import logging
import multiprocessing as mp
import os
import pathlib
import sys
import traceback
from time import sleep
from typing import Any, Generator, Optional

from modyn.common.benchmark import Stopwatch
from modyn.supervisor.internal.evaluation_result_writer import LogResultWriter
from modyn.supervisor.internal.grpc.enums import CounterAction, IdType, MsgType, PipelineStage
from modyn.supervisor.internal.grpc.template_msg import counter_submsg, dataset_submsg, id_submsg, pipeline_stage_msg
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.triggers import Trigger
from modyn.utils import dynamic_module_import

logger = logging.getLogger(__name__)
EXCEPTION_EXITCODE = 8


class PipelineExecutor:
    def __init__(
        self,
        start_timestamp: int,
        pipeline_id: int,
        modyn_config: dict,
        pipeline_config: dict,
        eval_directory: str,
        supervisor_supported_eval_result_writers: dict,
        pipeline_status_queue: mp.Queue,
        training_status_queue: mp.Queue,
        eval_status_queue: mp.Queue,
        start_replay_at: Optional[int] = None,
        stop_replay_at: Optional[int] = None,
        maximum_triggers: Optional[int] = None,
    ) -> None:
        self.stage = PipelineStage.INIT

        self.start_timestamp = start_timestamp
        self.pipeline_id = pipeline_id
        self.modyn_config = modyn_config
        self.pipeline_config = pipeline_config
        self.eval_directory = pathlib.Path(eval_directory)
        self.supervisor_supported_eval_result_writers = supervisor_supported_eval_result_writers
        self.pipeline_status_queue = pipeline_status_queue
        self.training_status_queue = training_status_queue
        self.eval_status_queue = eval_status_queue

        self.previous_model_id: Optional[int] = None
        if self.pipeline_config["training"]["initial_model"] == "pretrained":
            self.previous_model_id = self.pipeline_config["training"]["initial_model_id"]

        self.grpc = GRPCHandler(
            self.modyn_config, self.pipeline_status_queue, self.training_status_queue, self.eval_status_queue
        )

        self.start_replay_at = start_replay_at
        self.stop_replay_at = stop_replay_at
        self.maximum_triggers = maximum_triggers

        self._sw = Stopwatch()
        self._pipeline_log_file = self.eval_directory / f"pipeline_{self.pipeline_id}.log"
        self.pipeline_log: dict[str, Any] = {
            "configuration": {"pipeline_config": self.pipeline_config, "modyn_config": self.modyn_config},
            "supervisor": {
                "triggers": {},
                "new_data_requests": [],
                "num_triggers": 0,
                "trigger_batch_times": [],
                "selector_informs": [],
            },
        }
        self._determine_pipeline_mode()
        self._setup_trigger()
        self._selector_batch_size = 128

        self.num_triggers = 0
        self.current_training_id: Optional[int] = None
        self.triggers: list[int] = []
        # this is to store the first and last timestamp of the remaining data after handling all triggers in
        # _handle_triggers_within_batch
        self.remaining_data_range: Optional[tuple[int, int]] = None

    def _update_pipeline_stage_and_enqueue_msg(
        self, stage: PipelineStage, msg_type: MsgType, submsg: Optional[dict[str, Any]] = None, log: bool = False
    ) -> None:
        self.stage = stage
        self.pipeline_status_queue.put(pipeline_stage_msg(self.stage, msg_type, submsg, log))

    def init_cluster_connection(self) -> None:
        self._update_pipeline_stage_and_enqueue_msg(PipelineStage.INIT_CLUSTER_CONNECTION, MsgType.GENERAL)
        self.grpc.init_cluster_connection()

    def _determine_pipeline_mode(self) -> None:
        if self.start_replay_at is None:
            self.pipeline_log["experiment"] = False
            self.experiment_mode = False
            if self.stop_replay_at is not None:
                raise ValueError("stop_replay_at can only be used in conjunction with start_replay_at.")
        else:
            self.pipeline_log["experiment"] = True
            self.pipeline_log["start_replay_at"] = self.start_replay_at
            self.pipeline_log["stop_replay_at"] = self.stop_replay_at
            self.experiment_mode = True

    def _setup_trigger(self) -> None:
        trigger_id = self.pipeline_config["trigger"]["id"]
        trigger_config = {}
        if "trigger_config" in self.pipeline_config["trigger"].keys():
            trigger_config = self.pipeline_config["trigger"]["trigger_config"]

        trigger_module = dynamic_module_import("modyn.supervisor.internal.triggers")
        self.trigger: Trigger = getattr(trigger_module, trigger_id)(trigger_config)
        self.trigger.init_trigger(self.pipeline_id, self.pipeline_config, self.modyn_config, self.eval_directory)
        if self.previous_model_id is not None:
            self.trigger.inform_previous_model(self.previous_model_id)

        assert self.trigger is not None, "Error during trigger initialization"

    def _persist_pipeline_log(self) -> None:
        if "PYTEST_CURRENT_TEST" in os.environ:
            json.dumps(self.pipeline_log)  # Enforce serialization to catch issues
            return  # But don't actually store in tests

        with open(self._pipeline_log_file, "w", encoding="utf-8") as logfile:
            json.dump(self.pipeline_log, logfile, indent=4)

    def get_dataset_selector_batch_size(self) -> None:
        # system configuration already validated, so the dataset_id will be present in the configuration file
        dataset_id = self.pipeline_config["data"]["dataset_id"]
        self._update_pipeline_stage_and_enqueue_msg(
            PipelineStage.GET_SELECTOR_BATCH_SIZE, MsgType.DATASET, dataset_submsg(dataset_id)
        )

        for dataset in self.modyn_config["storage"]["datasets"]:
            if dataset["name"] == dataset_id:
                if "selector_batch_size" in dataset:
                    self._selector_batch_size = dataset["selector_batch_size"]
                break

    def _handle_new_data(self, new_data: list[tuple[int, int, int]]) -> bool:
        """This function handles new data during experiments or actual pipeline execution.
        We partition `new_data` into batches of `selector_batch_size` to reduce selector latency in case of a trigger.
        If a data point within a batch causes a trigger,
        we inform the selector about all data points including that data point.
        Otherwise, the selector is informed
        """
        logger.info(f"Received {len(new_data)} new data points. Handling batches.")
        new_data.sort(key=lambda tup: tup[1])
        any_training_triggered = False
        new_data_len = len(new_data)
        self._update_pipeline_stage_and_enqueue_msg(
            PipelineStage.HANDLE_NEW_DATA,
            MsgType.COUNTER,
            counter_submsg(CounterAction.CREATE, {"new_data_len": new_data_len}),
        )

        for i in range(0, new_data_len, self._selector_batch_size):
            batch = new_data[i : i + self._selector_batch_size]
            batch_size = self._selector_batch_size if i + self._selector_batch_size < new_data_len else new_data_len - i
            self._update_pipeline_stage_and_enqueue_msg(
                PipelineStage.HANDLE_NEW_DATA,
                MsgType.COUNTER,
                counter_submsg(CounterAction.UPDATE, {"batch_size": batch_size}),
            )

            triggered = self._handle_new_data_batch(batch)
            any_training_triggered = any_training_triggered or triggered
            if self.maximum_triggers is not None and self.num_triggers >= self.maximum_triggers:
                logger.info(f"Reached trigger limit ({self.maximum_triggers}), exiting.")
                break

        self._update_pipeline_stage_and_enqueue_msg(
            PipelineStage.NEW_DATA_HANDLED, MsgType.COUNTER, counter_submsg(CounterAction.CLOSE)
        )

        return any_training_triggered

    def _handle_new_data_batch(self, batch: list[tuple[int, int, int]]) -> bool:
        self._sw.start("trigger_inform", overwrite=True)
        triggering_indices: Generator[int, None, None] = self.trigger.inform(batch)
        num_triggers = self._handle_triggers_within_batch(batch, triggering_indices)

        logger.info(f"There are {num_triggers} triggers in this batch.")
        self.pipeline_log["supervisor"]["num_triggers"] += num_triggers
        self.pipeline_log["supervisor"]["trigger_batch_times"].append(
            {"batch_size": len(batch), "time": self._sw.stop("trigger_inform"), "num_triggers": num_triggers}
        )

        if num_triggers == 0:
            self._sw.start("selector_inform", overwrite=True)
            selector_log = self.grpc.inform_selector(self.pipeline_id, batch)
            self.pipeline_log["supervisor"]["selector_informs"].append(
                {"total_selector_time": self._sw.stop(), "selector_log": selector_log}
            )

        return num_triggers > 0

    def _run_training(self, trigger_id: int) -> None:
        """Run training for trigger on GPU and block until done."""
        assert self.pipeline_id is not None, "_run_training called without a registered pipeline."
        self._update_pipeline_stage_and_enqueue_msg(
            PipelineStage.RUN_TRAINING, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id)
        )
        logger.info(f"Running training for trigger {trigger_id}")

        self._sw.start("train", overwrite=True)
        num_samples_to_pass_per_trigger = self.pipeline_config["training"].get("num_samples_to_pass", [])
        current_trigger_index = len(self.triggers)
        if current_trigger_index <= len(num_samples_to_pass_per_trigger) - 1:
            num_samples_to_pass = num_samples_to_pass_per_trigger[current_trigger_index]
        else:
            num_samples_to_pass = None

        self.current_training_id = self.grpc.start_training(
            self.pipeline_id, trigger_id, self.pipeline_config, self.previous_model_id, num_samples_to_pass
        )

        self.stage = PipelineStage.WAIT_FOR_TRAINING_COMPLETION
        trainer_log = self.grpc.wait_for_training_completion(self.current_training_id, self.pipeline_id, trigger_id)

        if trigger_id not in self.pipeline_log["supervisor"]["triggers"]:
            self.pipeline_log["supervisor"]["triggers"][trigger_id] = {}  # can happen in tests

        self.pipeline_log["supervisor"]["triggers"][trigger_id]["total_trainer_time"] = self._sw.stop()
        self.pipeline_log["supervisor"]["triggers"][trigger_id]["trainer_log"] = trainer_log

        self._update_pipeline_stage_and_enqueue_msg(
            PipelineStage.STORE_TRAINED_MODEL, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id)
        )
        # We store the trained model for evaluation in any case.
        self._sw.start("store_trained_model", overwrite=True)
        model_id = self.grpc.store_trained_model(self.current_training_id)
        self.trigger.inform_previous_model(model_id)
        self.pipeline_log["supervisor"]["triggers"][trigger_id]["store_trained_model_time"] = self._sw.stop()

        # Only if the pipeline actually wants to continue the training on it, we set previous model.
        if self.pipeline_config["training"]["use_previous_model"]:
            self.previous_model_id = model_id

        self.triggers.append(trigger_id)

        # Start evaluation
        if "evaluation" in self.pipeline_config:
            self._update_pipeline_stage_and_enqueue_msg(
                PipelineStage.EVALUATE, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id)
            )
            # TODO(#300) Add evaluator to pipeline log
            evaluations = self.grpc.start_evaluation(model_id, self.pipeline_config)
            self.grpc.wait_for_evaluation_completion(self.current_training_id, evaluations)

            self._update_pipeline_stage_and_enqueue_msg(
                PipelineStage.STORE_EVALUATION_RESULTS, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id)
            )
            writer_names: set[str] = set(self.pipeline_config["evaluation"]["result_writers"])
            writers = [self._init_evaluation_writer(name, trigger_id) for name in writer_names]
            self.grpc.store_evaluation_results(writers, evaluations)

    def _get_trigger_timespan(
        self, is_first_triggering_data: bool, triggering_data: list[tuple[int, int, int]]
    ) -> tuple[int, int]:
        if is_first_triggering_data:
            # now it is the first trigger in this batch. Triggering_data can be empty.
            # when it is indeed empty, then there is remaining data in the last batch
            # because num_samples_in_trigger is not 0.
            assert len(triggering_data) > 0 or self.remaining_data_range is not None

            if self.remaining_data_range is not None:
                first_timestamp = self.remaining_data_range[0]
                last_timestamp = self.remaining_data_range[1] if len(triggering_data) == 0 else triggering_data[-1][1]
            else:
                first_timestamp = triggering_data[0][1]
                last_timestamp = triggering_data[-1][1]
        else:
            assert len(triggering_data) > 0
            # since num_samples_in_trigger is not 0, we are sure that triggering_data is not empty
            first_timestamp = triggering_data[0][1]
            last_timestamp = triggering_data[-1][1]

        return first_timestamp, last_timestamp

    def _handle_triggers_within_batch(
        self, batch: list[tuple[int, int, int]], triggering_indices: Generator[int, None, None]
    ) -> int:
        previous_trigger_idx = 0
        logger.info("Handling triggers within batch.")
        self._update_pipeline_stage_and_enqueue_msg(PipelineStage.HANDLE_TRIGGERS_WITHIN_BATCH, MsgType.GENERAL)
        num_triggers = 0

        triggering_idx_list = []

        for i, triggering_idx in enumerate(triggering_indices):
            triggering_idx_list.append(triggering_idx)
            num_triggers += 1
            self._update_pipeline_stage_and_enqueue_msg(PipelineStage.INFORM_SELECTOR_AND_TRIGGER, MsgType.GENERAL)
            triggering_data = batch[previous_trigger_idx : triggering_idx + 1]
            previous_trigger_idx = triggering_idx + 1

            # This call informs the selector about the data until (and including)
            # the data point that caused the trigger and then also notifies it about the triggering.
            # This means the next training call on trigger_id will guarantee
            # that all data until that point has been processed by the selector.
            self._sw.start("selector_inform", overwrite=True)
            trigger_id, selector_log = self.grpc.inform_selector_and_trigger(self.pipeline_id, triggering_data)
            self.pipeline_log["supervisor"]["triggers"][trigger_id] = {
                "total_selector_time": self._sw.stop(),
                "selector_log": selector_log,
            }
            self._persist_pipeline_log()

            num_samples_in_trigger = self.grpc.get_number_of_samples(self.pipeline_id, trigger_id)
            if num_samples_in_trigger > 0:
                self.trigger.inform_previous_trigger_and_data_points(trigger_id, num_samples_in_trigger)
                first_timestamp, last_timestamp = self._get_trigger_timespan(i == 0, triggering_data)
                self.pipeline_log["supervisor"]["triggers"][trigger_id]["first_timestamp"] = first_timestamp
                self.pipeline_log["supervisor"]["triggers"][trigger_id]["last_timestamp"] = last_timestamp
                # reset the remaining data range since we have processed it now
                self.remaining_data_range = None
                self._run_training(trigger_id)  # Blocks until training is done.
                self._update_pipeline_stage_and_enqueue_msg(
                    PipelineStage.HANDLE_TRIGGERS_WITHIN_BATCH, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id)
                )
            else:
                logger.info(f"Skipping training on empty trigger {trigger_id}]")

            self.num_triggers = self.num_triggers + 1
            if self.maximum_triggers is not None and self.num_triggers >= self.maximum_triggers:
                return num_triggers

        # we have to inform the Selector about the remaining data in this batch.
        if len(triggering_idx_list) == 0:
            remaining_data = batch
        else:
            remaining_data = batch[triggering_idx_list[-1] + 1 :]

        logger.info(f"There are {len(remaining_data)} data points remaining after the trigger.")
        if len(remaining_data) > 0:
            # These data points will be included in the next trigger
            # because we inform the Selector about them,
            # just like other batches with no trigger at all are included.
            self._sw.start("selector_inform", overwrite=True)
            selector_log = self.grpc.inform_selector(self.pipeline_id, remaining_data)
            self.pipeline_log["supervisor"]["selector_informs"].append(
                {"total_selector_time": self._sw.stop(), "selector_log": selector_log}
            )
            if self.remaining_data_range is not None:
                # extend the range from last time
                self.remaining_data_range = (self.remaining_data_range[0], remaining_data[-1][1])
            else:
                self.remaining_data_range = (remaining_data[0][1], remaining_data[-1][1])
        else:
            self.remaining_data_range = None

        self._persist_pipeline_log()
        return num_triggers

    def _init_evaluation_writer(self, name: str, trigger_id: int) -> LogResultWriter:
        return self.supervisor_supported_eval_result_writers[name](self.pipeline_id, trigger_id, self.eval_directory)

    def replay_data(self) -> None:
        assert self.start_replay_at is not None, "Cannot call replay_data when start_replay_at is None"
        dataset_id = self.pipeline_config["data"]["dataset_id"]
        self._update_pipeline_stage_and_enqueue_msg(
            PipelineStage.REPLAY_DATA, MsgType.DATASET, dataset_submsg(dataset_id)
        )
        logger.info("Starting data replay.")

        if self.stop_replay_at is None:
            generator = self.grpc.get_new_data_since(dataset_id, self.start_replay_at)
        else:
            generator = self.grpc.get_data_in_interval(dataset_id, self.start_replay_at, self.stop_replay_at)

        for replay_data, request_time in generator:
            assert isinstance(replay_data, list)
            assert isinstance(request_time, int)
            self.pipeline_log["supervisor"]["new_data_requests"].append(
                {"time": request_time, "num_items": len(replay_data)}
            )
            self._handle_new_data(replay_data)
            self._persist_pipeline_log()
            if self.maximum_triggers is not None and self.num_triggers >= self.maximum_triggers:
                logger.info("Exiting replay loop due to trigger limit.")
                break

        self._update_pipeline_stage_and_enqueue_msg(
            PipelineStage.REPLAY_DATA_DONE, MsgType.DATASET, dataset_submsg(dataset_id)
        )

    def shutdown_trainer(self) -> None:
        if self.current_training_id is not None:
            self.grpc.stop_training_at_trainer_server(self.current_training_id)

    def wait_for_new_data(self, start_timestamp: int) -> None:
        last_timestamp = start_timestamp
        dataset_id = self.pipeline_config["data"]["dataset_id"]

        previous_largest_keys = set()

        logger.info("Press CTRL+C at any time to shutdown the pipeline.")

        continue_running = True

        try:
            while continue_running:
                self._update_pipeline_stage_and_enqueue_msg(
                    PipelineStage.FETCH_NEW_DATA, MsgType.DATASET, dataset_submsg(dataset_id)
                )

                trigger_occured = False
                largest_keys = set()
                for new_data, _ in self.grpc.get_new_data_since(dataset_id, last_timestamp):
                    # Since get_new_data_since is inclusive, we need to filter out the keys
                    # we have already processed in the previous get_new_data_since request
                    new_data = [
                        (key, timestamp, label)
                        for (key, timestamp, label) in new_data
                        if key not in previous_largest_keys
                    ]
                    last_timestamp = (
                        max((timestamp for (_, timestamp, _) in new_data)) if len(new_data) > 0 else last_timestamp
                    )

                    # Remember all data points with last_timestamp so we do not process them again in the next iteration
                    # We use a set to have a O(1) check in the line above.
                    largest_keys.update({key for (key, timestamp, _) in new_data if timestamp == last_timestamp})

                    if self._handle_new_data(new_data):
                        trigger_occured = True

                    if self.maximum_triggers is not None and self.num_triggers >= self.maximum_triggers:
                        continue_running = False

                previous_largest_keys = largest_keys
                if not trigger_occured:
                    self._update_pipeline_stage_and_enqueue_msg(
                        PipelineStage.WAIT_FOR_NEW_DATA, MsgType.DATASET, dataset_submsg(dataset_id)
                    )
                    sleep(2)

        except KeyboardInterrupt:
            logger.info("Initiating shutdown.")
            self.shutdown_trainer()
            logger.info("Shutdown successful.")

    def execute(self) -> None:
        logger.info(f"[pipeline {self.pipeline_id}] Get dataset selector batch size.")
        self.get_dataset_selector_batch_size()

        logger.info(f"[pipeline {self.pipeline_id}] Start executing, experiment mode {self.experiment_mode}.")
        if self.experiment_mode:
            self.replay_data()
        else:
            self.wait_for_new_data(self.start_timestamp)

        logger.info(f"[pipeline {self.pipeline_id}] Execution done. Persist log.")
        self._update_pipeline_stage_and_enqueue_msg(PipelineStage.DONE, MsgType.GENERAL)
        self._persist_pipeline_log()


# pylint: disable=too-many-locals
def execute_pipeline(
    start_timestamp: int,
    pipeline_id: int,
    modyn_config: dict,
    pipeline_config: dict,
    eval_directory: str,
    supervisor_supported_eval_result_writers: dict,
    exception_queue: mp.Queue,
    pipeline_status_queue: mp.Queue,
    training_status_queue: mp.Queue,
    eval_status_queue: mp.Queue,
    start_replay_at: Optional[int] = None,
    stop_replay_at: Optional[int] = None,
    maximum_triggers: Optional[int] = None,
) -> None:
    try:
        pipeline = PipelineExecutor(
            start_timestamp,
            pipeline_id,
            modyn_config,
            pipeline_config,
            eval_directory,
            supervisor_supported_eval_result_writers,
            pipeline_status_queue,
            training_status_queue,
            eval_status_queue,
            start_replay_at,
            stop_replay_at,
            maximum_triggers,
        )
        pipeline.init_cluster_connection()
        pipeline.execute()
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        exception_queue.put(exception_msg)
        sys.exit(EXCEPTION_EXITCODE)
