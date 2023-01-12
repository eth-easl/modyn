import logging
import os
import pathlib
import typing
from time import sleep

from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from modyn.backend.supervisor.internal.trigger import Trigger
from modyn.utils import (
    dynamic_module_import,
    model_available,
    trigger_available,
    validate_yaml,
)

logger = logging.getLogger(__name__)


class Supervisor:

    # TODO(#63): Get these from the Trainer and Selector, as soon as that functionality is merged.
    supported_strategies: list[str] = ["finetune"]
    supported_initial_models: list[str] = ["random"]

    def __init__(
        self,
        pipeline_config: dict,
        modyn_config: dict,
        start_replay_at: typing.Optional[int] = None,
        stop_replay_at: typing.Optional[int] = None,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.modyn_config = modyn_config
        self.current_training_id = None

        if not self.validate_pipeline_config():
            raise ValueError("Invalid pipeline configuration")

        logging.info("Setting up connections to cluster components.")
        self.grpc = GRPCHandler(modyn_config)

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
        strategy = self.pipeline_config["training"]["strategy"]
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

        if strategy == "finetune":
            if (
                "strategy_config" not in self.pipeline_config["training"].keys()
                or "limit" not in self.pipeline_config["training"]["strategy_config"].keys()
            ):
                logger.warning("Did not give any explicit limit on finetuning strategy. Assuming no limit.")

        if initial_model not in Supervisor.supported_initial_models:
            logger.error(
                f"Unsupported initial model: {initial_model}."
                f"Supported initial models = {Supervisor.supported_initial_models}"
            )
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
            self.grpc.shutdown_trainer_server(self.current_training_id)

    def wait_for_new_data(self, start_timestamp: int) -> None:
        last_timestamp = start_timestamp
        dataset_id = self.pipeline_config["data"]["dataset_id"]

        last_keys = set()

        logger.info("Press CTRL+C at any time to shutdown the pipeline.")

        try:
            while True:
                new_data = self.grpc.get_new_data_since(dataset_id, last_timestamp)
                # Since get_new_data_since is inclusive, we need to filter out the keys we have already processed
                new_data = [(key, timestamp) for (key, timestamp) in new_data if key not in last_keys]
                last_timestamp = max([timestamp for (_, timestamp) in new_data]) if len(new_data) > 0 else last_timestamp

                # Remember all data points with last_timestamp so we do not process them again in the next iteration
                last_keys = set([key for (key, timestamp) in new_data if timestamp == last_timestamp])

                if not self._handle_new_data(new_data):
                    sleep(2)

        except KeyboardInterrupt:
            logger.info("Initiating shutdown.")
            self.shutdown_trainer()
            logger.info("Shutdown successful.")

    def _handle_new_data(self, new_data: list[tuple[str, int]], selector_batch_size: int = 128) -> bool:
        """ This function handles new data during experiments or actual pipeline execution.
            We partition `new_data` into batches of `selector_batch_size` to reduce selector latency in case of a trigger.
            If a data point within a batch causes a trigger, we inform the selector about all data points including that data point.
            Otherwise, the selector is informed
        """
        new_data.sort(key=lambda tup: tup[1])
        any_training_triggered = False

        for i in range(0, len(new_data), selector_batch_size):
            batch = new_data[i : i + selector_batch_size]
            triggered = self._handle_new_data_batch(batch)
            any_training_triggered = any_training_triggered or triggered

        return any_training_triggered

    def _handle_new_data_batch(self, batch: list[tuple[str, int]]) -> bool:
        triggering_indices = self.trigger.inform(batch) 

        if len(triggering_indices) > 0:
            self._handle_triggers_within_batch(batch, triggering_indices)
            return True
        else:
            self.grpc.inform_selector(self.pipeline_id, batch)
            return False

    def _handle_triggers_within_batch(self, batch: list[tuple[str, int]], triggering_indices: list[int]) -> None:
        previous_trigger_idx = 0
        for i, triggering_idx in enumerate(triggering_indices):
            triggering_data = batch [previous_trigger_idx : triggering_idx + 1]
            previous_trigger_idx = triggering_idx + 1

            # This call informs the selector about the data until (and including) the data point that caused the trigger and then also notifies it about the triggering.
            # This means the next training call on trigger_id will guarantee that all data until that point has been processed by the selector.
            trigger_id = self.grpc.inform_selector_and_trigger(self.pipeline_id, triggering_data)
            self._run_training(trigger_id) # Blocks until training is done.

            # If no other trigger is coming in this batch, we have to inform the Selector about the remaining data in this batch.
            if i == len(triggering_indices) - 1:
                remaining_data = batch [ triggering_idx + 1 : ]

                if len(remaining_data) > 0:
                    # These data points will be included in the next trigger because we inform the Selector about them, just like other batches with no trigger at all are included.
                    self.grpc.inform_selector(self.pipeline_id, remaining_data)


    def _run_training(self, trigger_id: int) -> None:
        """Run training for trigger on GPU and block until done.
        """
        assert self.pipeline_id is not None, "Callback called without a registered pipeline."
        self.current_training_id = self.grpc.start_trainer_server(self.pipeline_id, trigger_id, self.pipeline_config)

        self.grpc.wait_for_training_completion(self.current_training_id)

    def initial_pass(self) -> None:
        # TODO(##10): Implement initial pass.
        # for reference = interval, fetch all data in the interval between start_timestamp and end_timestamp
        # for reference = amount, we need support from the storage module to return the required keys
        pass

    def replay_data(self) -> None:
        assert self.start_replay_at is not None, "Cannot call replay_data when start_replay_at is None"
        dataset_id = self.pipeline_config["data"]["dataset_id"]

        if self.stop_replay_at is None:
            replay_data = self.grpc.get_new_data_since(dataset_id, self.start_replay_at)
        else:
            replay_data = self.grpc.get_data_in_interval(dataset_id, self.start_replay_at, self.stop_replay_at)

        self._handle_new_data(replay_data)

    def end_pipeline(self) -> None:
        # deregister etc
        pass

    def pipeline(self) -> None:
        start_timestamp = self.grpc.get_time_at_storage()
        self.pipeline_id = self.grpc.register_pipeline_at_selector(self.pipeline_config)

        self.initial_pass()

        if self.experiment_mode:
            self.replay_data()
        else:
            self.wait_for_new_data(start_timestamp)

        self.grpc.unregister_pipeline_at_selector(self.pipeline_id)
