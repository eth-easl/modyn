from __future__ import annotations

# TODO (THIS PR)!!! check history and reimplement changes (take ours merge happened here)

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep
from typing import Any, Callable, Optional

from modyn.common.benchmark import Stopwatch
from modyn.supervisor.internal.evaluation_result_writer import LogResultWriter
from modyn.supervisor.internal.grpc.enums import (
    CounterAction,
    IdType,
    MsgType,
    PipelineStage,
)
from modyn.supervisor.internal.grpc.template_msg import (
    counter_submsg,
    dataset_submsg,
    id_submsg,
    pipeline_stage_msg,
)
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.triggers import Trigger
from modyn.utils import dynamic_module_import

from .models import PipelineOptions

logger = logging.getLogger(__name__)
EXCEPTION_EXITCODE = 8

# TODO: pipeline hierarchy abstraction e.g. main pipeline, new data batch pipeline, eval pipeline, ...
@dataclass
class _RegisteredStage:
    """Represent a registered pipeline stage that includes a callable function and the next stage."""
    
    stage: PipelineStage
    func: Callable
    next: PipelineStage | None = None
    """If next stage if None, the next stage will be decided by the return value of the current stage's `func`.
    
    If both are None, the pipeline ends.
    """

# registered dynamically with decorators
stages: dict[PipelineStage, _RegisteredStage] = {}
"""Dynamically registered pipeline stages."""

# TODO subpipeline
def register_stage(stage: PipelineStage, *, next: PipelineStage | None = None, logging: bool = True) -> Callable:
    """Decorator to register a pipeline stage handler function."""
    # TODO: disable logging parameter
    # TODO: 
    
    def wrapper_outer(func: Callable[[Any,], PipelineStage | None]) -> Callable[[Any,], PipelineStage | None]:
        assert stage not in stages
        stages[stage] = (PipelineStage(
            stage=stage,
            func=func,
            next=next
        ))
        def wrapper(*args: Any, **kwargs: Any) -> PipelineStage | None:
            return func(*args, **kwargs)
        return wrapper
    return wrapper_outer


@dataclass
class ExecutionState:
    """Represent the state of the pipeline executor."""
        
    grpc: GRPCHandler

    _pipeline_log_file: Path
    pipeline_log: dict[str, Any]
    
    previous_model_id: Optional[int] = None
    _selector_batch_size = 128

    num_triggers = 0
    current_training_id: Optional[int] = None
    trained_models: list[int] = []
    triggers: list[int] = []
    
    sw = field(default_factory=Stopwatch)
    
    # TODO: pipeline artifacts

class PipelineExecutor:
    def __init__(self, options: PipelineOptions) -> None:
        self.stage = PipelineStage.INIT
        self.options = options

    # -------------------------------------------- Pipeline orchestration -------------------------------------------- #
        
    def execute(self) -> None:
        self.get_dataset_selector_batch_size()

        logger.info(f"[pipeline {self.options.pipeline_id}] Start executing, experiment mode {self.experiment_mode}.")
        if self.experiment_mode:
            self.replay_data()

            if self.options.evaluation_matrix:
                self.build_evaluation_matrix()
        else:
            self.wait_for_new_data(self.options.start_timestamp)

        logger.info(f"[pipeline {self.options.pipeline_id}] Execution done. Persist log.")
        self.options.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.DONE, MsgType.GENERAL))
        self._persist_pipeline_log()
        
    
        
    def execute(self) -> None:
        """Coordinates all pipelines stages until the pipeline execution is finished.
        
        Measures the time for each stage and logs the pipeline state.
        """
        
        print("executor")
        self.stage = PipelineStage.INIT
        while True:
            stage = stages[self.stage]
            
            # TODO log time and start timer
            returned_stage = stage.func(self)
            # TODO stop time and report duration (or end time)
            
            if returned_stage and stage.next:
                raise RuntimeError("A pipeline stage must either pre-define a next stage or return a stage. Not both!")
            if stage.next:
                self.stage = stage.next
            elif returned_stage:
                self.stage = returned_stage
            else:
                print("Pipeline execution finished.")
                return

    # ------------------------------------------------ Pipeline stages ----------------------------------------------- #
    
    # These functions are not suppose to be called manually.

    # Setup
    @register_stage(PipelineStage.INIT, next=PipelineStage.INIT_CLUSTER_CONNECTION)
    def _init(self) -> None:
        self.previous_model_id: Optional[int] = None
        if self.options.pipeline_config["training"]["initial_model"] == "pretrained":
            self.previous_model_id = self.options.pipeline_config["training"]["initial_model_id"]

        self.grpc = GRPCHandler(
            self.options.modyn_config, self.options.pipeline_status_queue, self.options.training_status_queue, self.options.eval_status_queue
        )

        self._sw = Stopwatch()
        self._pipeline_log_file = self.options.eval_directory / f"pipeline_{self.options.pipeline_id}.log"
        self.pipeline_log: dict[str, Any] = {
            "configuration": {"pipeline_config": self.options.pipeline_config, "modyn_config": self.options.modyn_config},
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

    @register_stage(PipelineStage.INIT_CLUSTER_CONNECTION, next=PipelineStage.GET_SELECTOR_BATCH_SIZE)
    def _init_cluster_connection(self) -> None:
        self.options.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.INIT_CLUSTER_CONNECTION, MsgType.GENERAL))
        self.grpc.init_cluster_connection()

    # GET_SELECTOR_BATCH_SIZE
    @register_stage(PipelineStage.GET_SELECTOR_BATCH_SIZE, next=None)
    def _get_dataset_selector_batch_size(self) -> PipelineStage:
        # system configuration already validated, so the dataset_id will be present in the configuration file
        dataset_id = self.options.pipeline_config["data"]["dataset_id"]
        self.options.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.GET_SELECTOR_BATCH_SIZE, MsgType.DATASET, dataset_submsg(dataset_id))
        )

        for dataset in self.options.modyn_config["storage"]["datasets"]:
            if dataset["name"] == dataset_id:
                if "selector_batch_size" in dataset:
                    self._selector_batch_size = dataset["selector_batch_size"]
                break

    @register_stage(PipelineStage._FORK_DATA_STRATEGY, logging=False)
    def _fork_data_strategy(self) -> PipelineStage:
        return PipelineStage.FETCH_NEW_DATA if self.experiment_mode else PipelineStage.REPLAY_DATA

    # Replay Data
    
    @register_stage(PipelineStage.REPLAY_DATA, next=PipelineStage.REPLAY_DATA_DONE)
    def replay_data(self) -> None:
        assert self.options.start_replay_at is not None, "Cannot call replay_data when start_replay_at is None"
        dataset_id = self.options.pipeline_config["data"]["dataset_id"]
        self.options.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.REPLAY_DATA, MsgType.DATASET, dataset_submsg(dataset_id))
        )
        logger.info("Starting data replay.")

        if self.options.stop_replay_at is None:
            generator = self.grpc.get_new_data_since(dataset_id, self.options.start_replay_at)
        else:
            generator = self.grpc.get_data_in_interval(dataset_id, self.options.start_replay_at, self.options.stop_replay_at)

        for replay_data, request_time in generator:
            assert isinstance(replay_data, list)
            assert isinstance(request_time, int)
            self.pipeline_log["supervisor"]["new_data_requests"].append(
                {"time": request_time, "num_items": len(replay_data)}
            )
            self.handle_new_data(replay_data)
            self._persist_pipeline_log()
            if self.options.maximum_triggers is not None and self.num_triggers >= self.options.maximum_triggers:
                logger.info("Exiting replay loop due to trigger limit.")
                break

        self.options.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.REPLAY_DATA_DONE, MsgType.DATASET, dataset_submsg(dataset_id))
        )

    @register_stage(PipelineStage.REPLAY_DATA_DONE, next=PipelineStage.DONE)
    def replay_data_done(self) -> None:
        # TODO: Implement this function
        pass

    # Wait for new data
    
    @register_stage(PipelineStage.FETCH_NEW_DATA, next=PipelineStage.WAIT_FOR_NEW_DATA)
    def fetch_new_data(self) -> None:
        # TODO: Implement this function
        pass
    
    @register_stage(PipelineStage.WAIT_FOR_NEW_DATA)
    def wait_for_new_data(self, start_timestamp: int) -> PipelineStage:
        last_timestamp = start_timestamp
        dataset_id = self.options.pipeline_config["data"]["dataset_id"]

        previous_largest_keys = set()

        logger.info("Press CTRL+C at any time to shutdown the pipeline.")

        continue_running = True

        try:
            while continue_running:
                self.options.pipeline_status_queue.put(
                    pipeline_stage_msg(PipelineStage.FETCH_NEW_DATA, MsgType.DATASET, dataset_submsg(dataset_id))
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

                    if self.handle_new_data(new_data):
                        trigger_occured = True

                    if self.options.maximum_triggers is not None and self.num_triggers >= self.options.maximum_triggers:
                        continue_running = False

                previous_largest_keys = largest_keys
                if not trigger_occured:
                    self.options.pipeline_status_queue.put(
                        pipeline_stage_msg(PipelineStage.WAIT_FOR_NEW_DATA, MsgType.DATASET, dataset_submsg(dataset_id))
                    )
                    sleep(2)

        except KeyboardInterrupt:
            logger.info("Initiating shutdown.")
            self._shutdown_trainer()
            logger.info("Shutdown successful.")

    # Process new data
    
    @register_stage(PipelineStage.HANDLE_NEW_DATA, next=PipelineStage.HANDLE_TRIGGERS_WITHIN_BATCH)
    def handle_new_data(self, new_data: list[tuple[int, int, int]]) -> bool:
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
        self.options.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.HANDLE_NEW_DATA,
                MsgType.COUNTER,
                counter_submsg(CounterAction.CREATE, {"new_data_len": new_data_len})
            )
        )

        for i in range(0, new_data_len, self._selector_batch_size):
            batch = new_data[i : i + self._selector_batch_size]
            batch_size = self._selector_batch_size if i + self._selector_batch_size < new_data_len else new_data_len - i
            self.options.pipeline_status_queue.put(
                pipeline_stage_msg(
                    PipelineStage.HANDLE_NEW_DATA,
                    MsgType.COUNTER,
                    counter_submsg(CounterAction.UPDATE, {"batch_size": batch_size})
                )
            )

            triggered = self._handle_new_data_batch(batch)
            any_training_triggered = any_training_triggered or triggered
            if self.options.maximum_triggers is not None and self.num_triggers >= self.options.maximum_triggers:
                logger.info(f"Reached trigger limit ({self.options.maximum_triggers}), exiting.")
                break

        self.options.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.NEW_DATA_HANDLED, MsgType.COUNTER, counter_submsg(CounterAction.CLOSE)log)
        )

        return any_training_triggered

    @register_stage(PipelineStage.HANDLE_TRIGGERS_WITHIN_BATCH)
    def handle_triggers_within_batch(self) -> PipelineStage:
        # TODO: Implement this function
        pass
    
    @register_stage(PipelineStage.INFORM_SELECTOR_AND_TRIGGER, next=PipelineStage.INFORM_SELECTOR_REMAINING_DATA)
    def inform_selector_and_trigger(self) -> None:
        # TODO: Implement this function
        pass
    
    # TODO: maybe unneeded
    @register_stage(PipelineStage._FORK_DECIDE_ANOTHER_TRIGGER)
    def _fork_decide_another_trigger(self) -> PipelineStage:
        # TODO: Implement this function
        pass
    
    @register_stage(PipelineStage.INFORM_SELECTOR_REMAINING_DATA, next=PipelineStage.NEW_DATA_HANDLED)
    def inform_selector_remaining_data(self) -> None:
        # TODO: Implement this function
        pass

    # Training

    @register_stage(PipelineStage.RUN_TRAINING, next=PipelineStage.WAIT_FOR_TRAINING_COMPLETION)
    def run_training(self, trigger_id: int) -> None:
        """Run training for trigger on GPU and block until done."""
        assert self.options.pipeline_id is not None, "_run_training called without a registered pipeline."
        self.options.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.RUN_TRAINING, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )
        logger.info(f"Running training for trigger {trigger_id}")

        self._sw.start("train", overwrite=True)
        num_samples_to_pass_per_trigger = self.options.pipeline_config["training"].get("num_samples_to_pass", [])
        current_trigger_index = len(self.triggers)
        if current_trigger_index <= len(num_samples_to_pass_per_trigger) - 1:
            num_samples_to_pass = num_samples_to_pass_per_trigger[current_trigger_index]
        else:
            num_samples_to_pass = None

        self.current_training_id = self.grpc.start_training(
            self.options.pipeline_id, trigger_id, self.options.pipeline_config, self.previous_model_id, num_samples_to_pass
        )

        self.stage = PipelineStage.WAIT_FOR_TRAINING_COMPLETION
        # TODO: split up into separate pipeline stages
        trainer_log = self.grpc.wait_for_training_completion(self.current_training_id, self.options.pipeline_id, trigger_id)

        if trigger_id not in self.pipeline_log["supervisor"]["triggers"]:
            self.pipeline_log["supervisor"]["triggers"][trigger_id] = {}  # can happen in tests

        self.pipeline_log["supervisor"]["triggers"][trigger_id]["total_trainer_time"] = self._sw.stop()
        self.pipeline_log["supervisor"]["triggers"][trigger_id]["trainer_log"] = trainer_log

        self.options.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.STORE_TRAINED_MODEL, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )
        
        # We store the trained model for evaluation in any case.
        self._sw.start("store_trained_model", overwrite=True)
        model_id = self.grpc.store_trained_model(self.current_training_id)
        self.pipeline_log["supervisor"]["triggers"][trigger_id]["store_trained_model_time"] = self._sw.stop()

        # Only if the pipeline actually wants to continue the training on it, we set previous model.
        if self.options.pipeline_config["training"]["use_previous_model"]:
            self.previous_model_id = model_id

        self.trained_models.append(model_id)
        self.triggers.append(trigger_id)

        # Start evaluation
        if "evaluation" in self.options.pipeline_config and not self.options.evaluation_matrix:
            self.options.pipeline_status_queue.put(
                pipeline_stage_msg(PipelineStage.EVALUATE, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
            )
            
            # TODO(#300) Add evaluator to pipeline log
            evaluations = self.grpc.start_evaluation(model_id, self.options.pipeline_config)
            self.grpc.wait_for_evaluation_completion(self.current_training_id, evaluations)

            self.options.pipeline_status_queue.put(
                pipeline_stage_msg(PipelineStage.STORE_EVALUATION_RESULTS, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
            )

            writer_names: set[str] = set(self.options.pipeline_config["evaluation"]["result_writers"])
            writers = [self._init_evaluation_writer(name, trigger_id) for name in writer_names]
            self.grpc.store_evaluation_results(writers, evaluations)

    @register_stage(PipelineStage.WAIT_FOR_TRAINING_COMPLETION, next=PipelineStage.TRAINING_COMPLETED)
    def wait_for_training_completion(self) -> None:
        # TODO: Implement this function
        # TODO: external trigger pipeline step
        pass
    
    @register_stage(PipelineStage.TRAINING_COMPLETED, next=PipelineStage.STORE_TRAINED_MODEL)
    def training_completed(self) -> None:
        # TODO: Implement this function
        pass
    
    @register_stage(PipelineStage.STORE_TRAINED_MODEL, next=PipelineStage._FORK_DECIDE_EVALUATION)
    def store_trained_model(self) -> None:
        # TODO: Implement this function
        pass
    
    @register_stage(PipelineStage._FORK_DECIDE_EVALUATION, logging=False)
    def _fork_decide_evaluation(self) -> PipelineStage:
        # TODO: check implementation
        if self.options.evaluation_matrix:
            return PipelineStage.EVALUATE
        return PipelineStage.HANDLE_TRIGGERS_WITHIN_BATCH
        
    # Evaluation
    
    @register_stage(PipelineStage.EVALUATE, next=PipelineStage.WAIT_FOR_EVALUATION_COMPLETION)
    def evaluate(self) -> None:
        # TODO: Implement this function
        pass
        
    @register_stage(PipelineStage.WAIT_FOR_EVALUATION_COMPLETION, next=PipelineStage.EVALUATION_COMPLETED)
    def wait_for_evaluation_completed(self) -> None:
        # TODO: Implement this function
        pass
    
    # TODO incorporate into pipeline
    def build_evaluation_matrix(self) -> None:
        self.pipeline_log["evaluation_matrix"] = {}
        for model in self.trained_models:
            self.pipeline_log["evaluation_matrix"][model] = {}
            for trigger in self.triggers:
                logger.info(f"Evaluating model {model} on trigger {trigger} for matrix.")
                evaluations = self.grpc.start_evaluation(model, self.options.pipeline_config, self.options.pipeline_id, trigger)
                assert self.current_training_id
                self.grpc.wait_for_evaluation_completion(self.current_training_id, evaluations)
                eval_result_writer = self._init_evaluation_writer("log", trigger)
                self.grpc.store_evaluation_results([eval_result_writer], evaluations)
                self.pipeline_log["evaluation_matrix"][model][trigger] = eval_result_writer.results

    @register_stage(PipelineStage.EVALUATION_COMPLETED, next=PipelineStage.STORE_EVALUATION_RESULTS)
    def evaluation_completed(self) -> None:
        # TODO: Implement this function
        pass
    
    @register_stage(PipelineStage.STORE_EVALUATION_RESULTS, next=PipelineStage.HANDLE_TRIGGERS_WITHIN_BATCH)
    def store_evaluation_results(self) -> None:
        # TODO: Implement this function
        pass

    # Teardown

    # DONE
    @register_stage(PipelineStage.DONE, next=PipelineStage.EXIT)
    def done(self) -> None:
        # TODO: Implement this function
        pass
    
    # EXIT
    @register_stage(PipelineStage.EXIT, next=None)
    def exit(self) -> None:
        return None  # end of pipeline
    
    
    # ---------------------------------------------------- Helpers --------------------------------------------------- #
        
    # pipeline
        
    def _determine_pipeline_mode(self) -> None:
        if self.options.start_replay_at is None:
            self.pipeline_log["experiment"] = False
            self.experiment_mode = False
            if self.options.stop_replay_at is not None:
                raise ValueError("stop_replay_at can only be used in conjunction with start_replay_at.")
        else:
            self.pipeline_log["experiment"] = True
            self.pipeline_log["start_replay_at"] = self.options.start_replay_at
            self.pipeline_log["stop_replay_at"] = self.options.stop_replay_at
            self.experiment_mode = True

    def _setup_trigger(self) -> None:
        trigger_id = self.options.pipeline_config["trigger"]["id"]
        trigger_config = {}
        if "trigger_config" in self.options.pipeline_config["trigger"].keys():
            trigger_config = self.options.pipeline_config["trigger"]["trigger_config"]

        trigger_module = dynamic_module_import("modyn.supervisor.internal.triggers")
        self.trigger: Trigger = getattr(trigger_module, trigger_id)(trigger_config)

        assert self.trigger is not None, "Error during trigger initialization"

    def _persist_pipeline_log(self) -> None:
        if "PYTEST_CURRENT_TEST" in os.environ:
            json.dumps(self.pipeline_log)  # Enforce serialization to catch issues
            return  # But don't actually store in tests

        with open(self._pipeline_log_file, "w", encoding="utf-8") as logfile:
            json.dump(self.pipeline_log, logfile, indent=4)

    def _handle_new_data_batch(self, batch: list[tuple[int, int, int]]) -> bool:
        self._sw.start("trigger_inform", overwrite=True)
        triggering_indices = self.trigger.inform(batch)
        num_triggers = len(triggering_indices)
        self.pipeline_log["supervisor"]["num_triggers"] += len(triggering_indices)
        self.pipeline_log["supervisor"]["trigger_batch_times"].append(
            {"batch_size": len(batch), "time": self._sw.stop("trigger_inform"), "num_triggers": num_triggers}
        )

        if num_triggers > 0:
            logger.info(f"There are {num_triggers} triggers in this batch.")
            self._handle_triggers_within_batch(batch, triggering_indices)
            return True

        self._sw.start("selector_inform", overwrite=True)
        selector_log = self.grpc.inform_selector(self.options.pipeline_id, batch)
        self.pipeline_log["supervisor"]["selector_informs"].append(
            {"total_selector_time": self._sw.stop(), "selector_log": selector_log}
        )

        return False

    def _handle_triggers_within_batch(self, batch: list[tuple[int, int, int]], triggering_indices: list[int]) -> None:
        previous_trigger_idx = 0
        logger.info("Handling triggers within batch.")
        self.options.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.HANDLE_TRIGGERS_WITHIN_BATCH, MsgType.GENERAL)
        )

        for i, triggering_idx in enumerate(triggering_indices):
            self.options.pipeline_status_queue.put(
                pipeline_stage_msg(PipelineStage.INFORM_SELECTOR_AND_TRIGGER, MsgType.GENERAL)
            )
            triggering_data = batch[previous_trigger_idx : triggering_idx + 1]
            previous_trigger_idx = triggering_idx + 1

            # This call informs the selector about the data until (and including)
            # the data point that caused the trigger and then also notifies it about the triggering.
            # This means the next training call on trigger_id will guarantee
            # that all data until that point has been processed by the selector.
            self._sw.start("selector_inform", overwrite=True)
            trigger_id, selector_log = self.grpc.inform_selector_and_trigger(self.options.pipeline_id, triggering_data)
            self.pipeline_log["supervisor"]["triggers"][trigger_id] = {
                "total_selector_time": self._sw.stop(),
                "selector_log": selector_log,
            }
            self._persist_pipeline_log()

            num_samples_in_trigger = self.grpc.get_number_of_samples(self.options.pipeline_id, trigger_id)
            if num_samples_in_trigger > 0:
                self.run_training(trigger_id)  # Blocks until training is done.
                self.options.pipeline_status_queue.put(
                    pipeline_stage_msg(PipelineStage.HANDLE_TRIGGERS_WITHIN_BATCH, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
                )

            else:
                logger.info(f"Skipping training on empty trigger {trigger_id}]")

            # TODO break/return missing? Loop
            # If no other trigger is coming in this batch,
            # we have to inform the Selector about the remaining data in this batch.
            if i == len(triggering_indices) - 1:
                remaining_data = batch[triggering_idx + 1 :]
                logger.info(f"There are {len(remaining_data)} data points remaining after the trigger.")

                if len(remaining_data) > 0:
                    # These data points will be included in the next trigger
                    # because we inform the Selector about them,
                    # just like other batches with no trigger at all are included.
                    self.options.pipeline_status_queue.put(
                        pipeline_stage_msg(
                            PipelineStage.INFORM_SELECTOR_REMAINING_DATA, 
                            MsgType.ID, 
                            id_submsg(IdType.TRIGGER, trigger_id)
                        )
                    )

                    self._sw.start("selector_inform", overwrite=True)
                    selector_log = self.grpc.inform_selector(self.options.pipeline_id, remaining_data)
                    self.pipeline_log["supervisor"]["selector_informs"].append(
                        {"total_selector_time": self._sw.stop(), "selector_log": selector_log}
                    )

            self._persist_pipeline_log()

            self.num_triggers = self.num_triggers + 1
            if self.options.maximum_triggers is not None and self.num_triggers >= self.options.maximum_triggers:
                break

    def _init_evaluation_writer(self, name: str, trigger_id: int) -> LogResultWriter:
        return self.options.supervisor_supported_eval_result_writers[name](self.options.pipeline_id, trigger_id, self.options.eval_directory)

    def _shutdown_trainer(self) -> None:
        if self.current_training_id is not None:
            self.grpc.stop_training_at_trainer_server(self.current_training_id)

    # ------------------------------------------------------ Run ----------------------------------------------------- #


def execute_pipeline(options: PipelineOptions) -> None:
    try:
        PipelineExecutor(options).execute()
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        options.exception_queue.put(exception_msg)
        sys.exit(EXCEPTION_EXITCODE)
