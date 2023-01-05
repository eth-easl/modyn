import typing
import logging
import os
import pathlib
from modyn.utils import model_available, trigger_available, validate_yaml, current_time_millis, dynamic_module_import
from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from modyn.backend.supervisor.internal.trigger import Trigger
from time import sleep

logger = logging.getLogger(__name__)


class Supervisor():

    # TODO(#63): Get these from the Trainer and Selector, as soon as that functionality is merged.
    supported_strategies: list[str] = ["finetune"]
    supported_initial_models: list[str] = ["random"]

    def __init__(self, pipeline_config: dict, modyn_config: dict, replay_at: typing.Optional[int]) -> None:
        self.pipeline_config = pipeline_config
        self.modyn_config = modyn_config

        if not self.validate_pipeline_config():
            raise ValueError("Invalid pipeline configuration")

        logging.info("Setting up connections to cluster components.")
        self.grpc = GRPCHandler(modyn_config)

        if not self.validate_system():
            raise ValueError("Invalid system configuration")

        if replay_at is None:
            self.experiment_mode = False
        else:
            self.experiment_mode = True
            self.replay_at = replay_at

        self._setup_trigger()

    def _setup_trigger(self) -> None:
        trigger_id = self.pipeline_config["trigger"]["id"]
        trigger_config = {}
        if "trigger_config" in self.pipeline_config["trigger"].keys():
            trigger_config = self.pipeline_config["trigger"]["trigger_config"]

        trigger_module = dynamic_module_import('modyn.backend.supervisor.internal.triggers')
        self.trigger: Trigger = getattr(trigger_module, trigger_id)(self._on_trigger, trigger_config)

        assert self.trigger is not None, "Error during trigger initialization"

    def validate_pipeline_config_schema(self) -> bool:
        schema_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "pipeline-schema.yaml"
        valid_yaml, exception = validate_yaml(self.pipeline_config, schema_path)

        if not valid_yaml:
            logger.error(
                f"Error while validating pipeline configuration file for schema-compliance: {exception.message}")
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
            if "strategy_config" not in self.pipeline_config["training"].keys() \
                    or "limit" not in self.pipeline_config["training"]["strategy_config"].keys():
                logger.warning("Did not give any explicit limit on finetuning strategy. Assuming no limit.")

        if initial_model not in Supervisor.supported_initial_models:
            logger.error(
                f"Unsupported initial model: {initial_model}."
                f"Supported initial models = {Supervisor.supported_initial_models}")
            is_valid = False

        if self.pipeline_config["training"]["initial_pass"]["activated"]:
            reference = self.pipeline_config["training"]["initial_pass"]["reference"]
            if reference not in ('amount', 'timestamp'):
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

    # pylint: disable-next=unused-argument
    def _query_new_data_from_storage(self, last_query: int) -> list[tuple[str, int]]:
        """Fetches all new data point from storage that have been added since last_query.
        To be implemented as soon as storage is merged (#44/#11).

                Parameters:
                        last_query (int): Timestamp (utils.current_time_millis) of last query

                Returns:
                        result_data (list[tuple[str, int]]): List of tuples containing new samples.
                            There is one tuple per sample, containing the sample key (str) and the
                            timestamp of the sample (int).
        """
        return []

    def wait_for_new_data(self, start_timestamp: int) -> None:
        last_query = start_timestamp

        logger.info("Press CTRL+C at any time to shutdown the pipeline.")

        try:
            while True:
                new_data = self._query_new_data_from_storage(last_query)
                # TODO(MaxiBoether): Currently, we lose datapoints that come in between the beginning of the
                # query and the return of the query, because their timestamp will be < last_query.
                # Needs to be fixed together with clock synchronization between storage and supervisor.
                # Probably, we will need to use the timestamp at storage.

                last_query = current_time_millis()
                if not self.trigger.inform(new_data):
                    # If the information didn't trigger, wait 2 seconds before querying storage again.
                    sleep(2)

        except KeyboardInterrupt:
            logger.info("Initiating supervisor shutdown.")
            # This might happen during training! We need to coordinate the shutdown here.
            return

    def _on_trigger(self) -> None:
        """Function that gets called by the trigger. This should start training on the GPU node.
        To be implemented.
        """

    def initial_pass(self) -> None:
        # initial_data = self._query_new_data_from_storage(0)
        # then: remove all samples that are too new (e.g., that we want to replay on)
        pass

    def replay_data(self) -> None:
        replay_data = self._query_new_data_from_storage(self.replay_at)
        self.trigger.inform(replay_data)

    def end_pipeline(self) -> None:
        # deregister etc
        pass

    def pipeline(self) -> None:
        start_timestamp = current_time_millis()
        self.initial_pass()

        if self.experiment_mode:
            self.replay_data()
        else:
            self.wait_for_new_data(start_timestamp)

        self.end_pipeline()
