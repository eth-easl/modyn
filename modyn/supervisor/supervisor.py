import json
import logging
import os
import pathlib
from time import sleep
from typing import Any, Optional

import enlighten
from modyn.common.benchmark import Stopwatch
from modyn.supervisor.internal.evaluation_result_writer import (
    AbstractEvaluationResultWriter,
    JsonResultWriter,
    LogResultWriter,
    TensorboardResultWriter,
)
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.triggers import Trigger
from modyn.utils import dynamic_module_import, is_directory_writable, model_available, trigger_available, validate_yaml
from tensorboard import program

logger = logging.getLogger(__name__)


class Supervisor:
    # pylint: disable=too-many-instance-attributes
    # This is a core class and we require the attributes.

    # TODO(#63): Get these from Selector
    supported_strategies: list[str] = [
        "NewDataStrategy",
        "FreshnessSamplingStrategy",
        "CoresetStrategy",
    ]

    supported_evaluation_result_writers: dict = {
        "json": JsonResultWriter,
        "tensorboard": TensorboardResultWriter,
        "log": LogResultWriter,
    }

    def __init__(
        self,
        pipeline_config: dict,
        modyn_config: dict,
        eval_directory: pathlib.Path,
        start_replay_at: Optional[int] = None,
        stop_replay_at: Optional[int] = None,
        maximum_triggers: Optional[int] = None,
        evaluation_matrix: bool = False,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.modyn_config = modyn_config
        self.eval_directory = eval_directory
        self.maximum_triggers = maximum_triggers
        self.num_triggers = 0
        self.current_training_id: Optional[int] = None
        self.pipeline_id: Optional[int] = None
        self.previous_model_id: Optional[int] = None
        self.evaluation_matrix = evaluation_matrix
        self.trained_models: list[int] = []
        self.triggers: list[int] = []

        self.pipeline_log: dict[str, Any] = {
            "configuration": {"pipeline_config": pipeline_config, "modyn_config": modyn_config},
            "supervisor": {
                "triggers": {},
                "new_data_requests": [],
                "num_triggers": 0,
                "trigger_batch_times": [],
                "selector_informs": [],
            },
        }
        self._sw = Stopwatch()
        self._pipeline_log_file = eval_directory / "pipeline.log"

        self.progress_mgr = enlighten.get_manager()
        self.status_bar = self.progress_mgr.status_bar(
            status_format="Modyn{fill}Current Task: {demo}{fill}{elapsed}",
            color="bold_underline_bright_white_on_lightslategray",
            justify=enlighten.Justify.CENTER,
            demo="Initializing",
            autorefresh=True,
            min_delta=0.5,
        )

        if not self.validate_pipeline_config():
            raise ValueError("Invalid pipeline configuration")

        if not is_directory_writable(self.eval_directory):
            raise ValueError("No permission to write to the evaluation results directory.")

        logging.info("Setting up connections to cluster components.")
        self.grpc = GRPCHandler(modyn_config, self.progress_mgr, self.status_bar)

        if not self.validate_system():
            raise ValueError("Invalid system configuration")

        if start_replay_at is None:
            self.pipeline_log["experiment"] = False
            self.experiment_mode = False
            if stop_replay_at is not None:
                raise ValueError("stop_replay_at can only be used in conjunction with start_replay_at.")
        else:
            self.pipeline_log["experiment"] = True
            self.pipeline_log["start_replay_at"] = start_replay_at
            self.pipeline_log["stop_replay_at"] = stop_replay_at

            self.experiment_mode = True
            self.start_replay_at = start_replay_at
            self.stop_replay_at = stop_replay_at

        self._setup_trigger()

        self._selector_batch_size = 128

        if "seed" in pipeline_config["training"]:
            self.grpc.seed_selector(pipeline_config["training"]["seed"])

        if "tensorboard" in self.modyn_config:
            port = self.modyn_config["tensorboard"]["port"]
            self._run_tensorboard(port)
            logger.info(f"Starting up tensorboard on port {port}.")

    def _setup_trigger(self) -> None:
        trigger_id = self.pipeline_config["trigger"]["id"]
        trigger_config = {}
        if "trigger_config" in self.pipeline_config["trigger"].keys():
            trigger_config = self.pipeline_config["trigger"]["trigger_config"]

        trigger_module = dynamic_module_import("modyn.supervisor.internal.triggers")
        self.trigger: Trigger = getattr(trigger_module, trigger_id)(trigger_config)

        assert self.trigger is not None, "Error during trigger initialization"

    def validate_pipeline_config_schema(self) -> bool:
        schema_path = (
            pathlib.Path(os.path.abspath(__file__)).parent.parent / "config" / "schema" / "pipeline-schema.yaml"
        )
        valid_yaml, exception = validate_yaml(self.pipeline_config, schema_path)

        if not valid_yaml:
            logger.error(
                f"Error while validating pipeline configuration file for schema-compliance: {exception.message}"
            )
            logger.error(exception)
            return False

        return True

    def _validate_evaluation_options(self, evaluation_config: dict) -> bool:
        is_valid = True

        dataset_ids = [dataset["dataset_id"] for dataset in evaluation_config["datasets"]]
        if len(set(dataset_ids)) < len(dataset_ids):
            logger.error("Dataset ids must be unique in evaluation")
            is_valid = False

        if "result_writers" in evaluation_config:
            writer_names = set(evaluation_config["result_writers"])
            if diff := writer_names.difference(self.supported_evaluation_result_writers.keys()):
                logger.error(f"Found invalid evaluation result writers: {', '.join(diff)}.")
                is_valid = False

        for dataset in evaluation_config["datasets"]:
            batch_size = dataset["batch_size"]
            if batch_size < 1:
                logger.error(f"Invalid batch size: {batch_size}.")
                is_valid = False

            dataloader_workers = dataset["dataloader_workers"]
            if dataloader_workers < 1:
                logger.error(f"Invalid dataloader worker amount: {dataloader_workers}.")
                is_valid = False

        return is_valid

    # pylint: disable=too-many-branches
    def _validate_training_options(self) -> bool:
        is_valid = True
        batch_size = self.pipeline_config["training"]["batch_size"]
        dataloader_workers = self.pipeline_config["training"]["dataloader_workers"]
        strategy = self.pipeline_config["training"]["selection_strategy"]["name"]
        initial_model = self.pipeline_config["training"]["initial_model"]

        if self.pipeline_config["training"]["gpus"] != 1:
            logger.error("Currently, only single GPU training is supported.")
            is_valid = False

        if batch_size < 1:
            logger.error(f"Invalid batch size: {batch_size}.")
            is_valid = False

        if dataloader_workers < 1:
            logger.error(f"Invalid dataloader worker amount: {dataloader_workers}.")
            is_valid = False

        if strategy not in Supervisor.supported_strategies:
            logger.error(f"Unsupported strategy: {strategy}. Supported strategies = {Supervisor.supported_strategies}")
            is_valid = False

        if initial_model not in ["random", "pretrained"]:
            logger.error("Only random and pretrained initial models are supported.")
            is_valid = False

        if initial_model == "pretrained":
            if not self.pipeline_config["training"]["use_previous_model"]:
                logger.error(
                    "Cannot have use_previous_model == False and use a pretrained initial model."
                    "Initial model would get lost after first trigger."
                )
                is_valid = False

            if "initial_model_id" not in self.pipeline_config["training"]:
                logger.error("Initial model set to pretrained, but no initial_model_id given")
                is_valid = False
            else:
                self.previous_model_id = self.pipeline_config["training"]["initial_model_id"]

        if self.pipeline_config["training"]["initial_pass"]["activated"]:
            reference = self.pipeline_config["training"]["initial_pass"]["reference"]
            if reference not in ("amount", "timestamp"):
                logger.error(f"Invalid reference for initial pass: {reference} (valid are 'amount' or 'timestamp')")
                is_valid = False

            if reference == "amount":
                amount = self.pipeline_config["training"]["initial_pass"]["amount"]
                if float(amount) > 1.0 or float(amount) < 0:
                    logger.error(f"Invalid initial pass amount: {amount}")
                    is_valid = False

        return is_valid

    def validate_pipeline_config_content(self) -> bool:
        is_valid = self._validate_training_options()

        model_id = self.pipeline_config["model"]["id"]
        if not model_available(model_id):
            logger.error(f"Model {model_id} is not available within Modyn.")
            is_valid = False

        trigger_id = self.pipeline_config["trigger"]["id"]
        if not trigger_available(trigger_id):
            logger.error(f"Trigger {trigger_id} is not available within Modyn.")
            is_valid = False

        if "evaluation" in self.pipeline_config:
            is_valid = is_valid and self._validate_evaluation_options(self.pipeline_config["evaluation"])

        if self.evaluation_matrix:
            if "evaluation" not in self.pipeline_config:
                logger.error("Can only create evaluation matrix with evaluation section.")
                is_valid = False
            else:
                train_dataset_id = self.pipeline_config["data"]["dataset_id"]
                train_dataset_in_eval = any(
                    dataset["dataset_id"] == train_dataset_id
                    for dataset in self.pipeline_config["evaluation"]["datasets"]
                )
                if not train_dataset_in_eval:
                    # TODO(#335): Fix this. Clean up in general.
                    logger.error(
                        "To create the evaluation matrix, you need to specify"
                        f" how to evaluate the training dataset {train_dataset_id}"  # pylint: disable
                        " in the evaluation section of the pipeline."
                    )
                    is_valid = False

        return is_valid

    def validate_pipeline_config(self) -> bool:
        return self.validate_pipeline_config_schema() and self.validate_pipeline_config_content()

    def dataset_available(self) -> bool:
        dataset_id = self.pipeline_config["data"]["dataset_id"]
        available = self.grpc.dataset_available(dataset_id)

        if not available:
            logger.error(f"Dataset {dataset_id} not available at storage.")

        return available

    def get_dataset_selector_batch_size(self) -> None:
        # system configuration already validated, so the dataset_id will be present in the configuration file
        dataset_id = self.pipeline_config["data"]["dataset_id"]
        for dataset in self.modyn_config["storage"]["datasets"]:
            if dataset["name"] == dataset_id:
                if "selector_batch_size" in dataset:
                    self._selector_batch_size = dataset["selector_batch_size"]
                break

    def validate_system(self) -> bool:
        return self.dataset_available() and self.grpc.trainer_server_available()

    def _run_tensorboard(self, port: str) -> None:
        logging.getLogger("tensorboard").setLevel(logging.ERROR)
        logging.getLogger("MARKDOWN").setLevel(logging.ERROR)

        tensorboard = program.TensorBoard()
        tensorboard.configure(
            argv=[
                None,
                "--logdir",
                str(self.eval_directory),
                "--bind_all",
                "--port",
                port,
                "--window_title",
                "Modyn TensorBoard",
            ]
        )
        tensorboard.launch()
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

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
                self.status_bar.update(demo="Fetching new data")
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
                    self.status_bar.update(demo="Waiting for new data...")
                    sleep(2)

        except KeyboardInterrupt:
            logger.info("Initiating shutdown.")
            self.shutdown_trainer()
            logger.info("Shutdown successful.")

    def _handle_new_data(self, new_data: list[tuple[int, int, int]]) -> bool:
        """This function handles new data during experiments or actual pipeline execution.
        We partition `new_data` into batches of `selector_batch_size` to reduce selector latency in case of a trigger.
        If a data point within a batch causes a trigger,
        we inform the selector about all data points including that data point.
        Otherwise, the selector is informed
        """
        self.status_bar.update(demo="Handling new data")
        logger.info(f"Received {len(new_data)} new data points. Handling batches.")
        new_data.sort(key=lambda tup: tup[1])
        any_training_triggered = False
        new_data_len = len(new_data)

        pbar = self.progress_mgr.counter(
            total=new_data_len, desc=f"[Pipeline {self.pipeline_id}] Processing New Samples", unit="samples"
        )

        for i in range(0, new_data_len, self._selector_batch_size):
            batch = new_data[i : i + self._selector_batch_size]
            triggered = self._handle_new_data_batch(batch)
            self.status_bar.update(demo="Handling new data")
            any_training_triggered = any_training_triggered or triggered
            pbar.update(self._selector_batch_size if i < new_data_len - 1 else pbar.total - pbar.count)
            if self.maximum_triggers is not None and self.num_triggers >= self.maximum_triggers:
                logger.info(f"Reached trigger limit ({self.maximum_triggers}), exiting.")
                break

        self.status_bar.update(demo="New data handled")
        pbar.clear(flush=True)
        pbar.close(clear=True)

        return any_training_triggered

    def _handle_new_data_batch(self, batch: list[tuple[int, int, int]]) -> bool:
        self._sw.start("trigger_inform", overwrite=True)
        triggering_indices = self.trigger.inform(batch)
        num_triggers = len(triggering_indices)
        self.pipeline_log["supervisor"]["num_triggers"] += len(triggering_indices)
        self.pipeline_log["supervisor"]["trigger_batch_times"].append(
            {"batch_size": len(batch), "time": self._sw.stop("trigger_inform"), "num_triggers": num_triggers}
        )

        if num_triggers > 0:
            self.status_bar.update(demo="Handling triggers")
            logger.info(f"There are {num_triggers} triggers in this batch.")
            self._handle_triggers_within_batch(batch, triggering_indices)
            return True

        self._sw.start("selector_inform", overwrite=True)
        selector_log = self.grpc.inform_selector(self.pipeline_id, batch)
        self.pipeline_log["supervisor"]["selector_informs"].append(
            {"total_selector_time": self._sw.stop(), "selector_log": selector_log}
        )

        return False

    def _handle_triggers_within_batch(self, batch: list[tuple[int, int, int]], triggering_indices: list[int]) -> None:
        previous_trigger_idx = 0
        logger.info("Handling triggers within batch.")
        for i, triggering_idx in enumerate(triggering_indices):
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
                self.status_bar.update(demo="Training")
                self._run_training(trigger_id)  # Blocks until training is done.
            else:
                logger.info(f"Skipping training on empty trigger {trigger_id}]")
            self.status_bar.update(demo="Handling triggers")

            # If no other trigger is coming in this batch,
            # we have to inform the Selector about the remaining data in this batch.
            if i == len(triggering_indices) - 1:
                remaining_data = batch[triggering_idx + 1 :]
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

            self._persist_pipeline_log()

            self.num_triggers = self.num_triggers + 1
            if self.maximum_triggers is not None and self.num_triggers >= self.maximum_triggers:
                break

    def _run_training(self, trigger_id: int) -> None:
        """Run training for trigger on GPU and block until done."""
        assert self.pipeline_id is not None, "_run_training called without a registered pipeline."
        logger.info(f"Running training for trigger {trigger_id}")
        self._sw.start("train", overwrite=True)
        self.current_training_id = self.grpc.start_training(
            self.pipeline_id, trigger_id, self.pipeline_config, self.previous_model_id
        )
        trainer_log = self.grpc.wait_for_training_completion(self.current_training_id, self.pipeline_id, trigger_id)

        if trigger_id not in self.pipeline_log["supervisor"]["triggers"]:
            self.pipeline_log["supervisor"]["triggers"][trigger_id] = {}  # can happen in tests

        self.pipeline_log["supervisor"]["triggers"][trigger_id]["total_trainer_time"] = self._sw.stop()
        self.pipeline_log["supervisor"]["triggers"][trigger_id]["trainer_log"] = trainer_log

        # We store the trained model for evaluation in any case.
        self._sw.start("store_trained_model", overwrite=True)
        model_id = self.grpc.store_trained_model(self.current_training_id)
        self.pipeline_log["supervisor"]["triggers"][trigger_id]["store_trained_model_time"] = self._sw.stop()

        # Only if the pipeline actually wants to continue the training on it, we set previous model.
        if self.pipeline_config["training"]["use_previous_model"]:
            self.previous_model_id = model_id

        self.trained_models.append(model_id)
        self.triggers.append(trigger_id)

        # Start evaluation
        if "evaluation" in self.pipeline_config and not self.evaluation_matrix:
            # TODO(#300) Add evaluator to pipeline log
            evaluations = self.grpc.start_evaluation(model_id, self.pipeline_config)
            self.grpc.wait_for_evaluation_completion(self.current_training_id, evaluations)

            writer_names: set[str] = set(self.pipeline_config["evaluation"]["result_writers"])
            writers = [self._init_evaluation_writer(name, trigger_id) for name in writer_names]
            self.grpc.store_evaluation_results(writers, evaluations)

    def _init_evaluation_writer(self, name: str, trigger_id: int) -> AbstractEvaluationResultWriter:
        return self.supported_evaluation_result_writers[name](self.pipeline_id, trigger_id, self.eval_directory)

    def initial_pass(self) -> None:
        # TODO(#128): Implement initial pass.
        # for reference = interval, fetch all data in the interval between start_timestamp and end_timestamp
        # for reference = amount, we need support from the storage module to return the required keys
        # In case self.previous_model_id is set, respect and update!
        pass

    def replay_data(self) -> None:
        assert self.start_replay_at is not None, "Cannot call replay_data when start_replay_at is None"
        dataset_id = self.pipeline_config["data"]["dataset_id"]
        self.status_bar.update(demo="Replaying data")
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

        self.status_bar.update(demo="Replay done")

    def _persist_pipeline_log(self) -> None:
        if "PYTEST_CURRENT_TEST" in os.environ:
            json.dumps(self.pipeline_log)  # Enforce serialization to catch issues
            return  # But don't actually store in tests

        with open(self._pipeline_log_file, "w", encoding="utf-8") as logfile:
            json.dump(self.pipeline_log, logfile, indent=4)

    def build_evaluation_matrix(self) -> None:
        self.pipeline_log["evaluation_matrix"] = {}
        for model in self.trained_models:
            self.pipeline_log["evaluation_matrix"][model] = {}
            for trigger in self.triggers:
                logger.info(f"Evaluating model {model} on trigger {trigger} for matrix.")
                evaluations = self.grpc.start_evaluation(model, self.pipeline_config, self.pipeline_id, trigger)
                self.grpc.wait_for_evaluation_completion(self.current_training_id, evaluations)
                eval_result_writer: LogResultWriter = self._init_evaluation_writer("log", trigger)
                self.grpc.store_evaluation_results([eval_result_writer], evaluations)
                self.pipeline_log["evaluation_matrix"][model][trigger] = eval_result_writer.results

    def pipeline(self) -> None:
        start_timestamp = self.grpc.get_time_at_storage()
        self.pipeline_id = self.grpc.register_pipeline_at_selector(self.pipeline_config)
        self.status_bar.update(demo="Initial Pass")

        self.initial_pass()
        logger.info("Initial pass completed.")

        self.get_dataset_selector_batch_size()
        if self.experiment_mode:
            self.replay_data()

            if self.evaluation_matrix:
                self.build_evaluation_matrix()
        else:
            self.wait_for_new_data(start_timestamp)

        self.status_bar.update(demo="Cleanup")
        logger.info("Pipeline done, unregistering.")
        self.grpc.unregister_pipeline_at_selector(self.pipeline_id)
        self._persist_pipeline_log()
