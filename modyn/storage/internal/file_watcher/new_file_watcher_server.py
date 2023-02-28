import logging
from ctypes import c_bool
from multiprocessing import Process, Value
from typing import Any, Dict

from modyn.storage.internal.database.models import Dataset
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.file_watcher.new_file_watcher import run_new_file_watcher

logger = logging.getLogger(__name__)


class NewFileWatcherServer:
    def __init__(self, modyn_config: dict, should_stop: Any):  # See https://github.com/python/typeshed/issues/8799
        """Initialize the new file watcher server.

        Args:
            modyn_config (dict): Configuration of the modyn module.
            should_stop (Any): Value that indicates if the new file watcher should stop.
        """
        self.modyn_config = modyn_config
        self.__should_stop = should_stop
        self.__file_watchers: Dict[int, (Process, Any)] = {}

    def __manage_file_watchers(self) -> None:
        """Manage the file watchers.

        This method will check if there are file watchers that are not watching a dataset anymore. If that is the case,
        the file watcher will be stopped.
        """
        with StorageDatabaseConnection(self.modyn_config) as storage_database_connection:
            session = storage_database_connection.session
            dataset_ids = [dataset.dataset_id for dataset in session.query(Dataset).all()]

            for dataset_id in self.__file_watchers:
                if dataset_id not in dataset_ids:
                    logger.debug(f"Stopping file watcher for dataset {dataset_id}")
                    self.__stop_file_watcher(dataset_id)

            for dataset_id in dataset_ids:
                if dataset_id not in self.__file_watchers:
                    logger.debug(f"Starting file watcher for dataset {dataset_id}")
                    self.__start_file_watcher(dataset_id)
                elif self.__file_watchers[dataset_id][0].exitcode is not None:
                    logger.debug(
                        f"File watcher for dataset {dataset_id} has an exit code of "
                        + f"{self.__file_watchers[dataset_id][0].exitcode}"
                    )
                    self.__start_file_watcher(dataset_id)
                elif not self.__file_watchers[dataset_id][0].is_alive():
                    logger.debug(f"File watcher for dataset {dataset_id} is not alive anymore")
                    self.__start_file_watcher(dataset_id)

    def __start_file_watcher(self, dataset_id: int) -> None:
        """Start a file watcher.

        Args:
            dataset_id (int): ID of the dataset that should be watched.
        """
        should_stop = Value(c_bool, False)
        file_watcher = Process(target=run_new_file_watcher, args=(self.modyn_config, dataset_id, should_stop))
        file_watcher.start()
        self.__file_watchers[dataset_id] = (file_watcher, should_stop)

    def __stop_file_watcher(self, dataset_id: int) -> None:
        """Stop a file watcher.

        Args:
            dataset_id (int): ID of the dataset that should be watched.
        """
        self.__file_watchers[dataset_id][1] = True
        self.__file_watchers[dataset_id][0].terminate()
        self.__file_watchers[dataset_id][0].join()
        del self.__file_watchers[dataset_id]

    def run(self) -> None:
        """Run the new file watcher.

        Args:
            modyn_config (dict): Configuration of the modyn module.
            should_stop (Value): Value that indicates if the watcher should stop.
        """
        while not self.__should_stop.value:
            self.__manage_file_watchers()

        for dataset_id in self.__file_watchers:
            self.__stop_file_watcher(dataset_id)


def run_watcher_server(modyn_config: dict, should_stop: Any):
    """Run the new file watcher server.

    Args:
        modyn_config (dict): Configuration of the modyn module.
        should_stop (Value): Value that indicates if the watcher should stop.
    """
    new_file_watcher_server = NewFileWatcherServer(modyn_config, should_stop)
    new_file_watcher_server.run()
