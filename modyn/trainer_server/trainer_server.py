import os
import pathlib
from typing import Tuple

from modyn.trainer_server.internal.grpc.grpc_server import GRPCServer
from modyn.utils.utils import validate_yaml

class TrainerServer:
    def __init__(self, config: dict) -> None:
        self.config = config

        # valid, errors = self._validate_config()
        # if not valid:
        #     raise ValueError(f"Invalid configuration: {errors}")

    def _validate_config(self) -> Tuple[bool, list[str]]:
        schema_path = (
            pathlib.Path(os.path.abspath(__file__)).parent.parent / "config" / "schema" / "modyn_config_schema.yaml"
        )
        return validate_yaml(self.config, schema_path)

    def run(self) -> None:
        print("run!")
        with GRPCServer(self.config) as server:
            server.wait_for_termination()