import logging
import os
import pathlib
from typing import Tuple
import json

from modyn.utils import validate_yaml
from modyn.storage.internal.grpc.grpc_server import GRPCServer
from modyn.storage.internal.database.database_connection import DatabaseConnection

logger = logging.getLogger(__name__)


class Storage():
    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config

        valid, errors = self._validate_config()
        if not valid:
            raise ValueError(f"Invalid configuration: {errors}")

    def _validate_config(self) -> Tuple[bool, list[str]]:
        schema_path = pathlib.Path(os.path.abspath(__file__)).parent.parent \
            / "config" / "schema" / "modyn_config_schema.yaml"
        return validate_yaml(self.modyn_config, schema_path)

    def run(self) -> None:
        #  Create the database tables.
        with DatabaseConnection(self.modyn_config) as database:
            database.create_all()

            for dataset in self.modyn_config['storage']['datasets']:
                if not database.add_dataset(dataset['name'],
                                            dataset['base_path'],
                                            dataset['filesystem_wrapper_type'],
                                            dataset['file_wrapper_type'],
                                            dataset['description'],
                                            dataset['version'],
                                            json.dumps(dataset['file_wrapper_config'])):
                    raise ValueError(f"Failed to add dataset {dataset['name']}")

        #  Start the dataset watcher process in a different thread.

        #  Start the storage grpc server.
        with GRPCServer(self.modyn_config) as server:
            server.wait_for_termination()
