import typing
import logging
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import yaml
import os
import pathlib
from modyn.utils import model_available
from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler

logger = logging.getLogger(__name__)


class Supervisor():
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

    def validate_pipeline_config_schema(self) -> bool:
        # TODO(MaxiBoether): Actually write the schema.
        schema_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "pipeline-schema.yaml"
        assert schema_path.is_file(), "Did not find pipeline configuration schema."
        with open(schema_path, "r") as f:
            pipeline_schema = yaml.safe_load(f)

        try:
            validate(self.pipeline_config, pipeline_schema)
        except ValidationError as e:
            logger.error(f"Error while validating pipeline configuration file for schema-compliance: {e.message}")
            logger.error(e)
            return False

        return True

    def validate_pipeline_config_content(self) -> bool:
        model_id = self.pipeline_config["model"]["id"]
        if not model_available(model_id):
            logger.error(f"Model {model_id} is not available within Modyn.")
            return False

        # TODO(MaxiBoether): More checks.

        return True

    def validate_pipeline_config(self) -> bool:
        return self.validate_pipeline_config_schema() and self.validate_pipeline_config_content()

    def dataset_available(self) -> bool:
        dataset_id = self.pipeline_config["data"]["dataset_id"]
        available = self.grpc.dataset_available(dataset_id)

        if not available:
            logger.error(f"Dataset {dataset_id} not available at storage.")

        return available

    def trainer_available(self) -> bool:
        return True

    def validate_system(self) -> bool:
        return self.dataset_available() and self.trainer_available()

    def wait_for_new_data(self) -> None:
        pass

    def initial_pass(self) -> None:
        pass

    def replay_data(self) -> None:
        pass

    def end_pipeline(self) -> None:
        # deregister etc
        pass

    def pipeline(self) -> None:
        self.initial_pass()
        if self.experiment_mode:
            self.replay_data()
        else:
            # TODO(MaxiBoether): think about data coming in between initial pass and pulling. probably just pass timestamp before initial pass started to pull and then get notified about that data
            self.wait_for_new_data()

        self.end_pipeline()
