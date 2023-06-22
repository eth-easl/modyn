import os
import pathlib
import shutil
import tempfile
from typing import Optional, Tuple

from jsonschema import ValidationError
from modyn.evaluator.internal.grpc.evaluator_grpc_server import EvaluatorGRPCServer
from modyn.utils import validate_yaml


class Evaluator:
    def __init__(self, config: dict) -> None:
        self.config = config

        valid, errors = self._validate_config()
        if not valid:
            raise ValueError(f"Invalid configuration: {errors}")

        self.working_directory = pathlib.Path(tempfile.gettempdir()) / "modyn_evaluator"

        if self.working_directory.exists() and self.working_directory.is_dir():
            shutil.rmtree(self.working_directory)

        self.working_directory.mkdir()

    def _validate_config(self) -> Tuple[bool, Optional[ValidationError]]:
        schema_path = (
            pathlib.Path(os.path.abspath(__file__)).parent.parent / "config" / "schema" / "modyn_config_schema.yaml"
        )
        return validate_yaml(self.config, schema_path)

    def run(self) -> None:
        with EvaluatorGRPCServer(self.config, self.working_directory) as server:
            server.wait_for_termination()
        shutil.rmtree(self.working_directory)
