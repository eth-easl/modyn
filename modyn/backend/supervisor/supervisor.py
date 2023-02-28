import logging
import os
import pathlib
from time import sleep
from typing import Optional

import enlighten
from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from modyn.backend.supervisor.internal.trigger import Trigger
from modyn.utils import dynamic_module_import, model_available, trigger_available, validate_yaml
from modyn.utils.utils import current_time_millis

logger = logging.getLogger(__name__)


class Supervisor:
    # pylint: disable=too-many-instance-attributes
    # This is a core class and we require the attributes.

    # TODO(#63): Get these from Selector
    supported_strategies: list[str] = ["NewDataStrategy", "FreshnessSamplingStrategy"]

    def __init__(
        self,
        pipeline_config: dict,
        modyn_config: dict,
        start_replay_at: Optional[int] = None,
        stop_replay_at: Optional[int] = None,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.modyn_config = modyn_config
        self.current_training_id: Optional[int] = None
        self.pipeline_id: Optional[int] = None
        self.previous_model: Optional[pathlib.Path] = None

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

        logging.info("Setting up connections to cluster components.")
        self.grpc = GRPCHandler(modyn_config, self.progress_mgr, self.status_bar)

        if not self.validate_system():
            raise ValueError("Invalid system configuration")

        if start_replay_at is None:
            self.experiment_mode = False
            if stop_replay_at is not None:
                raise ValueError("stop_replay_at can only be used in conjunction with start_replay_at.")
        else:
            self.experiment_mode = True
            self.start_replay_at = start_replay_at
            self.stop_replay_at = stop_replay_at

        self._setup_trigger()
        self._setup_model_directory()

    def _setup_model_directory(self) -> None:
        self.model_storage_directory = (
            pathlib.Path(os.getcwd())
            / f"models_{self.pipeline_config['pipeline']['name'].replace(' ', '_')}"
            / str(current_time_millis())
        )
        os.makedirs(self.model_storage_directory)

    def _setup_trigger(self) -> None:
        trigger_id = self.pipeline_config["trigger"]["id"]
        trigger_config = {}
        if "trigger_config" in self.pipeline_config["trigger"].keys():
            trigger_config = self.pipeline_config["trigger"]["trigger_config"]

        trigger_module = dynamic_module_import("modyn.backend.supervisor.internal.triggers")
        self.trigger: Trigger = getattr(trigger_module, trigger_id)(trigger_config)

        assert self.trigger is not None, "Error during trigger initialization"

    def validate_pipeline_config_schema(self) -> bool:
        schema_path = (
            pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "schema" / "pipeline-schema.yaml"
        )
        valid_yaml, exception = validate_yaml(self.pipeline_config, schema_path)

        if not valid_yaml:
            logger.error(
                f"Error while validating pipeline configuration file for schema-compliance: {exception.message}"
            )
            logger.error(exception)
            return False

        return True

    def _validate_training_options(self) -> bool:
        is_valid = True
        batch_size = self.pipeline_config["training"]["batch_size"]
        strategy = self.pipeline_config["training"]["selection_strategy"]["name"]
        initial_model = self.pipeline_config["training"]["initial_model"]

        if self.pipeline_config["training"]["gpus"] != 1:
            logger.error("Currently, only single GPU training is supported.")
            is_valid = False

        if batch_size < 1:
            logger.error("Invalid batch size: {batch_size}")
            is_valid = False

        if strategy not in Supervisor.supported_strategies:
            logger.error(f"Unsupported strategy: {strategy}. Supported strategies = {Supervisor.supported_strategies}")
            is_valid = False

        if initial_model not in ["random", "pretrained"]:
            logger.error("Only random and pretrained initial models are supported.")
            is_valid = False

        if initial_model == "pretrained":
            if "initial_model_path" not in self.pipeline_config["training"]:
                logger.error("Initial model set to pretrained, but no initial_model_path given")
                is_valid = False
            else:
                self.previous_model = self.pipeline_config["training"]["initial_model_path"]
                assert self.previous_model is not None  # makes mypy happy
                if not self.previous_model.exists():
                    logger.error(f"Path {self.previous_model} does not exist.")
                    is_valid = False

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

        return is_valid

    def validate_pipeline_config(self) -> bool:
        return self.validate_pipeline_config_schema() and self.validate_pipeline_config_content()

    def dataset_available(self) -> bool:
        dataset_id = self.pipeline_config["data"]["dataset_id"]
        available = self.grpc.dataset_available(dataset_id)

        if not available:
            logger.error(f"Dataset {dataset_id} not available at storage.")

        return available

    def validate_system(self) -> bool:
        return self.dataset_available() and self.grpc.trainer_server_available()

    def shutdown_trainer(self) -> None:
        if self.current_training_id is not None:
            self.grpc.stop_training_at_trainer_server(self.current_training_id)

    def wait_for_new_data(self, start_timestamp: int) -> None:
        last_timestamp = start_timestamp
        dataset_id = self.pipeline_config["data"]["dataset_id"]

        previous_largest_keys = set()

        logger.info("Press CTRL+C at any time to shutdown the pipeline.")

        try:
            while True:
                self.status_bar.update(demo="Fetching new data")
                trigger_occured = False
                largest_keys = set()
                for new_data in self.grpc.get_new_data_since(dataset_id, last_timestamp):
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

                previous_largest_keys = largest_keys
                if not trigger_occured:
                    self.status_bar.update(demo="Waiting for new data...")
                    sleep(2)

        except KeyboardInterrupt:
            logger.info("Initiating shutdown.")
            self.shutdown_trainer()
            logger.info("Shutdown successful.")

    def _handle_new_data(self, new_data: list[tuple[str, int, int]], selector_batch_size: int = 128) -> bool:
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

        for i in range(0, new_data_len, selector_batch_size):
            batch = new_data[i : i + selector_batch_size]
            triggered = self._handle_new_data_batch(batch)
            self.status_bar.update(demo="Handling new data")
            any_training_triggered = any_training_triggered or triggered
            pbar.update(selector_batch_size if i < new_data_len - 1 else pbar.total - pbar.count)

        self.status_bar.update(demo="New data handled")
        pbar.clear(flush=True)
        pbar.close(clear=True)

        return any_training_triggered

    def _handle_new_data_batch(self, batch: list[tuple[str, int, int]]) -> bool:
        triggering_indices = self.trigger.inform(batch)

        if len(triggering_indices) > 0:
            self.status_bar.update(demo="Handling triggers")
            logger.info(f"There are {len(triggering_indices)} triggers in this batch.")
            self._handle_triggers_within_batch(batch, triggering_indices)
            return True

        self.grpc.inform_selector(self.pipeline_id, batch)
        return False

    def _handle_triggers_within_batch(self, batch: list[tuple[str, int, int]], triggering_indices: list[int]) -> None:
        previous_trigger_idx = 0
        logger.info("Handling triggers within batch.")
        for i, triggering_idx in enumerate(triggering_indices):
            triggering_data = batch[previous_trigger_idx : triggering_idx + 1]
            previous_trigger_idx = triggering_idx + 1

            # This call informs the selector about the data until (and including)
            # the data point that caused the trigger and then also notifies it about the triggering.
            # This means the next training call on trigger_id will guarantee
            # that all data until that point has been processed by the selector.
            trigger_id = self.grpc.inform_selector_and_trigger(self.pipeline_id, triggering_data)
            self.status_bar.update(demo="Training")
            self._run_training(trigger_id)  # Blocks until training is done.
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
                    self.grpc.inform_selector(self.pipeline_id, remaining_data)

    def _run_training(self, trigger_id: int) -> None:
        """Run training for trigger on GPU and block until done."""
        assert self.pipeline_id is not None, "_run_training called without a registered pipeline."
        logger.info(f"Running training for trigger {trigger_id}")

        self.current_training_id = self.grpc.start_training(
            self.pipeline_id, trigger_id, self.pipeline_config, self.previous_model
        )
        self.grpc.wait_for_training_completion(self.current_training_id, self.pipeline_id, trigger_id)

        self.previous_model = self.grpc.fetch_trained_model(self.current_training_id, self.model_storage_directory)

    def initial_pass(self) -> None:
        # TODO(#128): Implement initial pass.
        # for reference = interval, fetch all data in the interval between start_timestamp and end_timestamp
        # for reference = amount, we need support from the storage module to return the required keys
        # In case self.previous_model is set, respect and update!
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

        for replay_data in generator:
            self._handle_new_data(replay_data)

        self.status_bar.update(demo="Replay done")

    def pipeline(self) -> None:
        start_timestamp = self.grpc.get_time_at_storage()
        self.pipeline_id = self.grpc.register_pipeline_at_selector(self.pipeline_config)
        self.status_bar.update(demo="Initial Pass")

        self.initial_pass()
        logger.info("Initial pass completed.")

        if self.experiment_mode:
            self.replay_data()
        else:
            self.wait_for_new_data(start_timestamp)

        self.status_bar.update(demo="Cleanup")
        logger.info("Pipeline done, unregistering.")
        self.grpc.unregister_pipeline_at_selector(self.pipeline_id)
