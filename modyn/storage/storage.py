import logging
import os
import pathlib
from typing import List, Tuple
from multiprocessing import Process

from modyn.utils import validate_yaml
from modyn.storage.internal.grpc.grpc_server import GRPCServer
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.dataset_watcher import DatasetWatcher

logger = logging.getLogger(__name__)


class Storage():
    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config

        valid, errors = self._validate_config()
        if not valid:
            raise ValueError(f"Invalid configuration: {errors}")

    def _validate_config(self) -> Tuple[bool, List[str]]:
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
                                            dataset['version']):
                    raise ValueError(f"Failed to add dataset {dataset['name']}")

        #  Start the dataset watcher process in a different thread.
        dataset_watcher = Process(target=DatasetWatcher(self.modyn_config).run)
        dataset_watcher.start()

        #  Start the storage grpc server.
        with GRPCServer(self.modyn_config) as server:
            server.wait_for_termination()

        #  Close the dataset_watcher process. This will cause a ValueError to be raised
        #  in the dataset_watcher process (because it's still running), but we can ignore it.
        #  See https://docs.python.org/3/library/multiprocessing.html#:~:text=in%20version%203.7.-,close()%C2%B6,-Close%20the%20Process  # noqa
        try:
            dataset_watcher.close()
        except ValueError:
            pass
