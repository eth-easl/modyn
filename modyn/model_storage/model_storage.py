import os
import pathlib
from typing import Optional, Tuple

from jsonschema import ValidationError
from modyn.utils import validate_yaml


class ModelStorage:
    def __init__(self, config: dict) -> None:
        self.config = config

        valid, errors = self._validate_config()
        if not valid:
            raise ValueError(f"Invalid configuration: {errors}")

    def _validate_config(self) -> Tuple[bool, Optional[ValidationError]]:
        schema_path = (
            pathlib.Path(os.path.abspath(__file__)).parent.parent / "config" / "schema" / "modyn_config_schema.yaml"
        )
        return validate_yaml(self.config, schema_path)

    def run(self) -> None:
        print("Running Model Storage Component")
        # run the model storage component
