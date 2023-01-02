import logging
from threading import Thread
import os
import pathlib
from typing import List, Tuple

from modyn.utils import validate_yaml
from modyn.storage.internal.grpc.grpc_server import GRPCServer
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.seeker import Seeker

logger = logging.getLogger(__name__)


class Storage():
    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config

        valid, errors = self._validate_config()
        if not valid:
            raise ValueError(f"Invalid configuration: {errors}")

    def _validate_config(self) -> Tuple[bool, List[str]]:
        schema_path = pathlib.Path(os.path.join(os.getcwd(), 'modyn', 'config', 'schema', 'modyn_config_schema.yaml'))
        return validate_yaml(self.modyn_config, schema_path)

    def run(self) -> None:
        #  Create the database tables.
        with DatabaseConnection(self.modyn_config) as database:
            database.create_all()

            for dataset in self.modyn_config['storage']['datasets']:
                database.add_dataset(dataset['name'],
                                     dataset['base_path'],
                                     dataset['filesystem_wrapper_type'],
                                     dataset['file_wrapper_type'],
                                     dataset['description'],
                                     dataset['version'])

        #  Start the seeker process in a different thread.
        seeker = Thread(target=Seeker(self.modyn_config).run)
        seeker.start()

        #  Start the storage grpc server.
        with GRPCServer(self.modyn_config) as server:
            server.wait_for_termination()

        seeker.join()
