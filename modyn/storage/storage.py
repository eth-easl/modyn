"""Storage module.

The storage module contains all classes and functions related to the retrieval of data from the
various storage backends.
"""

import logging
import os
import pathlib
from typing import Tuple
import json
from multiprocessing import Value, Process
from ctypes import c_bool


from modyn.utils import validate_yaml
from modyn.storage.internal.grpc.grpc_server import GRPCServer
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.new_file_watcher import run_watcher

logger = logging.getLogger(__name__)


class Storage:
    """Storage server.

    The storage server is responsible for the retrieval of data from the various storage backends.
    """

    def __init__(self, modyn_config: dict) -> None:
        """Initialize the storage server.

        Args:
            modyn_config (dict): Configuration of the modyn module.

        Raises:
            ValueError: Invalid configuration.
        """
        self.modyn_config = modyn_config

        valid, errors = self._validate_config()
        if not valid:
            raise ValueError(f"Invalid configuration: {errors}")

    def _validate_config(self) -> Tuple[bool, list[str]]:
        schema_path = (
            pathlib.Path(os.path.abspath(__file__)).parent.parent / "config" / "schema" / "modyn_config_schema.yaml"
        )
        return validate_yaml(self.modyn_config, schema_path)

    def run(self) -> None:
        """Run the storage server.

        Raises:
            ValueError: Failed to add dataset.
        """
        #  Create the database tables.
        with DatabaseConnection(self.modyn_config) as database:
            database.create_all()

            for dataset in self.modyn_config["storage"]["datasets"]:
                if not database.add_dataset(
                    dataset["name"],
                    dataset["base_path"],
                    dataset["filesystem_wrapper_type"],
                    dataset["file_wrapper_type"],
                    dataset["description"],
                    dataset["version"],
                    json.dumps(dataset["file_wrapper_config"]),
                ):
                    raise ValueError(f"Failed to add dataset {dataset['name']}")

        #  Start the dataset watcher process in a different thread.
        should_stop = Value(c_bool, False)
        watcher_process = Process(target=run_watcher, args=(self.modyn_config, should_stop))
        watcher_process.start()

        #  Start the storage grpc server.
        with GRPCServer(self.modyn_config) as server:
            server.wait_for_termination()

        should_stop.value = True  # type: ignore  # See https://github.com/python/typeshed/issues/8799
        watcher_process.join()
