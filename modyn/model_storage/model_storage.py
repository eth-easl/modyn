import os
import pathlib
from typing import Optional, Tuple

from jsonschema import ValidationError
from modyn.common.ftp.ftp_server import FTPServer
from modyn.model_storage.internal.grpc.grpc_server import GRPCServer
from modyn.utils import validate_yaml


class ModelStorage:
    def __init__(self, config: dict) -> None:
        self.config = config

        valid, errors = self._validate_config()
        if not valid:
            raise ValueError(f"Invalid configuration: {errors}")

        self._setup_model_storage_directory()

    def _validate_config(self) -> Tuple[bool, Optional[ValidationError]]:
        schema_path = (
            pathlib.Path(os.path.abspath(__file__)).parent.parent / "config" / "schema" / "modyn_config_schema.yaml"
        )
        return validate_yaml(self.config, schema_path)

    def _setup_model_storage_directory(self) -> None:
        self.model_storage_directory = pathlib.Path(os.getcwd()) / "model_storage"
        os.makedirs(self.model_storage_directory)

    def run(self) -> None:
        with GRPCServer(self.config, self.model_storage_directory) as server:
            with FTPServer(self.config["model_storage"]["ftp_port"], self.model_storage_directory):
                server.wait_for_termination()
