import logging
import os
import pathlib
import typing
from time import sleep

from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from modyn.backend.supervisor.internal.trigger import Trigger
from modyn.utils import (
    current_time_millis,
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
        start_replay_at: typing.Optional[int],
        stop_replay_at: typing.Optional[int],
    ) -> None:
        self.pipeline_config = pipeline_config
        self.modyn_config = modyn_config

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
        self.trigger: Trigger = getattr(trigger_module, trigger_id)(self._on_trigger, trigger_config)

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

    def trainer_available(self) -> bool:
        # TODO(MaxiBoether): implement
        return True

    def validate_system(self) -> bool:
        return self.dataset_available() and self.trainer_available()

    def shutdown_training(self) -> None:
        # TODO(MaxiBoether): implement
        pass

    def wait_for_new_data(self, start_timestamp: int, training_id: int) -> None:
        last_timestamp = start_timestamp
        dataset_id = self.pipeline_config["data"]["dataset_id"]

        logger.info("Press CTRL+C at any time to shutdown the pipeline.")

        try:
            while True:
                # TODO(MaxiBoether): is get_new_data_since inclusive or exclusive? If inclusive, we need to filter out the data points we have already seen
                # i.e., last_timestamp will get set to 42, and then if we receive all keys with 42 again, we need to remove the keys that we saw in the last iteration
                # If it is exclusive, we might lose new data with the same timestamp that came in while the gRPC request was being processed
                # Best solution would probably be inclusive + filtering out the keys that we processed at the supervisor.
                new_data = self.grpc.get_new_data_since(dataset_id, last_timestamp)

                last_timestamp = max([timestamp for (_, timestamp) in new_data])
                if not self.trigger.inform(new_data):
                    sleep(2)

        except KeyboardInterrupt:
            logger.info("Initiating shutdown.")
            self.shutdown_training()
            logger.info("Shutdown successful.")

    def _on_trigger(self, triggering_key: str, key_timestamp: int) -> None:
        """Function that gets called by the trigger. This should inform the selector, start training on the GPU node and block until it has finished.
        To be implemented.
        """

    def initial_pass(self, training_id: int) -> None:
        # initial_data = self._query_new_data_from_storage(0)
        # then: remove all samples that are too new (e.g., that we want to replay on)
        pass

    def replay_data(self, training_id: int) -> None:
        # TODO(MaxiBoether): Think about inclusivity/exclusivity of get_data functions
        assert self.start_replay_at is not None, "Cannot call replay_data when start_replay_at is None"
        dataset_id = self.pipeline_config["data"]["dataset_id"]

        if self.stop_replay_at is None:
            replay_data = self.grpc.get_new_data_since(dataset_id, self.start_replay_at)
        else:
            replay_data = self.grpc.get_data_in_interval(dataset_id, self.start_replay_at, self.stop_replay_at)

        self.trigger.inform(replay_data)

    def end_pipeline(self, training_id: int) -> None:
        # deregister etc
        pass

    def register_training(self, start_timestamp: int) -> int:
        # register at selector and gpu node, check what foteini has done here
        # returns training identifier

        # at selector, we should inform it about the start_timestamp so that the first trigger can be handeled correctly
        # if self.experimentmode, inform selector about self.start_replay_at instead.
        pass

    def pipeline(self) -> None:
        start_timestamp = self.grpc.get_time_at_storage()
        training_id = self.register_training(start_timestamp)

        self.initial_pass(training_id)

        if self.experiment_mode:
            self.replay_data(training_id)
        else:
            self.wait_for_new_data(start_timestamp, training_id)

        self.end_pipeline(training_id)
