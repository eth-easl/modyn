import typing
import logging
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import yaml
import os
import pathlib
from modyn.utils import model_available

logger = logging.getLogger(__name__)

class Supervisor():
    def __init__(self, pipeline_config: dict, modyn_config: dict, replay_at: typing.Optional[int]) -> None:
        if self.validate_pipeline_config(pipeline_config):
            self.pipeline_config = pipeline_config
        else:
            raise ValueError("Invalid pipeline configuration")

        if self.validate_system(modyn_config):
            self.modyn_config = modyn_config
        else:
            raise ValueError("Invalid system configuration")

        if replay_at is None:
            self.experiment_mode = False
            self.setup_data_pulling()
        else:
            self.experiment_mode = True
            self.replay_at = replay_at

    def validate_pipeline_config_schema(self, pipeline_config: dict) -> bool:
        schema_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "pipeline-schema.yaml"
        assert schema_path.is_file(), "Did not find pipeline configuration schema."
        with open(schema_path, "r") as f:
            pipeline_schema = yaml.safe_load(f)

        try:
            validate(pipeline_config, pipeline_schema)
        except ValidationError as e:
            logger.error(f"Error while validating pipeline configuration file for schema-compliance: {e.message}")
            logger.error(e)
            return False

        return True

    def validate_pipeline_config_content(self, pipeline_config: dict) -> bool:
        model_id = pipeline_config["model"]["id"]
        if not model_available(model_id):
            logger.error(f"Model {model_id} is not available within Modyn.")
            return False

        # TODO: More checks.

        return True

    def validate_pipeline_config(self, pipeline_config: dict) -> bool:
        return self.validate_pipeline_config_schema(pipeline_config) and self.validate_pipeline_config_content(pipeline_config)

    def validate_system(self, modyn_config: dict) -> bool:
        return True

    def setup_data_pulling(self) -> None:
        pass

    def initial_pass(self) -> None:
        pass

    def pipeline(self) -> None:
        self.initial_pass()