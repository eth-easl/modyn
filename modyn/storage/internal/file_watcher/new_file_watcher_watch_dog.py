import logging
import time
from ctypes import c_bool
from multiprocessing import Process, Value
from typing import Any

from modyn.storage.internal.database.models import Dataset
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.file_watcher.new_file_watcher import run_new_file_watcher

logger = logging.getLogger(__name__)


class NewFileWatcherWatchDog:
    def __init__(self, modyn_config: dict, should_stop: Any):  # See https://github.com/python/typeshed/issues/8799
        """Initialize the new file watcher watch dog.

        Args:
            modyn_config (dict): Configuration of the modyn module.
            should_stop (Any): Value that indicates if the new file watcher should stop.
        """
        self.modyn_config = modyn_config
        self.__should_stop = should_stop
        self._file_watcher_processes: dict[int, tuple[Process, Any, int]] = {}

    def _manage_file_watcher_processes(self) -> None:
        """Manage the file watchers.

        This method will check if there are file watchers that are not watching a dataset anymore. If that is the case,
        the file watcher will be stopped.
        """
        with StorageDatabaseConnection(self.modyn_config) as storage_database_connection:
            session = storage_database_connection.session
            dataset_ids = [dataset.dataset_id for dataset in session.query(Dataset).all()]
            dataset_ids_in_file_watcher_processes = list(self._file_watcher_processes.keys())
            for dataset_id in dataset_ids_in_file_watcher_processes:
                if dataset_id not in dataset_ids:
                    logger.debug(f"Stopping file watcher for dataset {dataset_id}")
                    self._stop_file_watcher_process(dataset_id)

            for dataset_id in dataset_ids:
                if dataset_id not in self._file_watcher_processes:
                    logger.debug(f"Starting file watcher for dataset {dataset_id}")
                    self._start_file_watcher_process(dataset_id)
                if self._file_watcher_processes[dataset_id][2] > 3:
                    logger.debug(f"Stopping file watcher for dataset {dataset_id} because it was restarted too often.")
                    self._stop_file_watcher_process(dataset_id)
                elif not self._file_watcher_processes[dataset_id][0].is_alive():
                    logger.debug(f"File watcher for dataset {dataset_id} is not alive. Restarting it.")
                    self._start_file_watcher_process(dataset_id)
                    self._file_watcher_processes[dataset_id] = (
                        self._file_watcher_processes[dataset_id][0],
                        self._file_watcher_processes[dataset_id][1],
                        self._file_watcher_processes[dataset_id][2] + 1,
                    )

    def _start_file_watcher_process(self, dataset_id: int) -> None:
        """Start a file watcher.

        Args:
            dataset_id (int): ID of the dataset that should be watched.
        """
        should_stop = Value(c_bool, False)
        file_watcher = Process(target=run_new_file_watcher, args=(self.modyn_config, dataset_id, should_stop))
        file_watcher.start()
        self._file_watcher_processes[dataset_id] = (file_watcher, should_stop, 0)

    def _stop_file_watcher_process(self, dataset_id: int) -> None:
        """Stop a file watcher.

        Args:
            dataset_id (int): ID of the dataset that should be watched.
        """
        self._file_watcher_processes[dataset_id][1].value = True
        i = 0
        while self._file_watcher_processes[dataset_id][0].is_alive() and i < 10:  # Wait for the file watcher to stop.
            time.sleep(1)
            i += 1
        if self._file_watcher_processes[dataset_id][0].is_alive():
            logger.debug(f"File watcher for dataset {dataset_id} is still alive. Terminating it.")
            self._file_watcher_processes[dataset_id][0].terminate()
        self._file_watcher_processes[dataset_id][0].join()
        del self._file_watcher_processes[dataset_id]

    def run(self) -> None:
        """Run the new file watcher watchdog.

        Args:
            modyn_config (dict): Configuration of the modyn module.
            should_stop (Value): Value that indicates if the watcher should stop.
        """
        while not self.__should_stop.value:
            self._manage_file_watcher_processes()
            time.sleep(3)

        for dataset_id in self._file_watcher_processes:
            self._stop_file_watcher_process(dataset_id)


def run_watcher_watch_dog(modyn_config: dict, should_stop: Any):  # type: ignore  # See https://github.com/python/typeshed/issues/8799  # noqa: E501
    """Run the new file watcher watch dog.

    Args:
        modyn_config (dict): Configuration of the modyn module.
        should_stop (Value): Value that indicates if the watcher should stop.
    """
    new_file_watcher_watch_dog = NewFileWatcherWatchDog(modyn_config, should_stop)
    new_file_watcher_watch_dog.run()
