"""New file watcher."""

import io
import json
import logging
import multiprocessing as mp
import os
import pathlib
import platform
import time
from typing import Any, Optional

import pandas as pd
from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.database.storage_database_utils import get_file_wrapper, get_filesystem_wrapper
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.utils import current_time_millis
from sqlalchemy import exc
from sqlalchemy.orm import exc as orm_exc
from sqlalchemy.orm.session import Session

logger = logging.getLogger(__name__)


class NewFileWatcher:
    """New file watcher.

    This class is responsible for watching all the filesystems of the datasets for new files. If a new file is found, it
    will be added to the database.
    """

    def __init__(
        self, modyn_config: dict, dataset_id: int, should_stop: Any
    ):  # See https://github.com/python/typeshed/issues/8799
        """Initialize the new file watcher.

        Args:
            modyn_config (dict): Configuration of the modyn module.
            should_stop (Any): Value that indicates if the new file watcher should stop.
        """
        self.modyn_config = modyn_config
        self.__should_stop = should_stop
        self.__dataset_id = dataset_id

        self._insertion_threads = modyn_config["storage"]["insertion_threads"]
        self._sample_dbinsertion_batchsize = modyn_config["storage"]["sample_dbinsertion_batchsize"] # todo add to schema

        self._is_test = "PYTEST_CURRENT_TEST" in os.environ
        self._is_mac = platform.system() == "Darwin"
        self._disable_mt = self._insertion_threads <= 0

        # Initialize dataset partition on Sample table
        with StorageDatabaseConnection(self.modyn_config) as database:
            Sample.add_dataset(self.__dataset_id, database.session, database.engine)

    def _seek(self, storage_database_connection: StorageDatabaseConnection, dataset: Dataset) -> None:
        """Seek the filesystem for all the datasets for new files and add them to the database.

        If last timestamp is not ignored, the last timestamp of the dataset will be used to only
        seek for files that have a timestamp that is equal or greater than the last timestamp.
        """
        if dataset is None:
            logger.warning(
                f"Dataset {self.__dataset_id} not found. Shutting down file watcher for dataset {self.__dataset_id}."
            )
            self.__should_stop.value = True
            return
        session = storage_database_connection.session
        try:
            logger.debug(
                f"Seeking for files in dataset {dataset.dataset_id} with a timestamp that \
                is equal or greater than {dataset.last_timestamp}"
            )
            self._seek_dataset(session, dataset)
            last_timestamp = (
                session.query(File.updated_at)
                .filter(File.dataset_id == dataset.dataset_id)
                .order_by(File.updated_at.desc())
                .first()
            )
            if last_timestamp is not None:
                session.query(Dataset).filter(Dataset.dataset_id == dataset.dataset_id).update(
                    {"last_timestamp": last_timestamp[0]}
                )
                session.commit()
        except orm_exc.ObjectDeletedError as error:
            # If the dataset was deleted, we should stop the file watcher and delete all the
            # orphaned files and samples
            logger.warning(
                f"Dataset {self.__dataset_id} was deleted. Shutting down "
                + f"file watcher for dataset {self.__dataset_id}. Error: {error}"
            )
            session.rollback()
            storage_database_connection.delete_dataset(dataset.name)
            self.__should_stop.value = True

    def _seek_dataset(self, session: Session, dataset: Dataset) -> None:
        """Seek the filesystem for a dataset for new files and add them to the database.

        If last timestamp is not ignored, the last timestamp of the dataset will be used to
        only seek for files that have a timestamp that is equal or greater than the last timestamp.

        Args:
            session (Session): Database session.
            dataset (Dataset): Dataset to seek.
        """
        filesystem_wrapper = get_filesystem_wrapper(dataset.filesystem_wrapper_type, dataset.base_path)

        if filesystem_wrapper.exists(dataset.base_path):
            if filesystem_wrapper.isdir(dataset.base_path):
                self._update_files_in_directory(
                    filesystem_wrapper,
                    dataset.file_wrapper_type,
                    dataset.base_path,
                    dataset.last_timestamp,
                    session,
                    dataset,
                )
            else:
                logger.critical(f"Path {dataset.base_path} is not a directory.")
        else:
            logger.warning(f"Path {dataset.base_path} does not exist.")

    def _get_datasets(self, session: Session) -> list[Dataset]:
        """Get all datasets."""
        datasets: Optional[list[Dataset]] = session.query(Dataset).all()

        if datasets is None or len(datasets) == 0:
            logger.warning("No datasets found.")
            return []

        return datasets

    @staticmethod
    def _file_unknown(session: Session, file_path: str) -> bool:
        """Check if a file is unknown.

        TODO (#147): This is a very inefficient way to check if a file is unknown. It should be replaced
        by a more efficient method.
        """
        return session.query(File).filter(File.path == file_path).first() is None

    @staticmethod
    def _postgres_copy_insertion(
        process_id: int, dataset_id: int, curr_df: pd.DataFrame, time_spent: dict, session: Session
    ) -> None:
        session_setup_start = current_time_millis()

        conn = session.connection().engine.raw_connection()
        cursor = conn.cursor()

        table_name = f"samples__did{dataset_id}"
        table_columns = "(dataset_id,file_id,index,label)"
        cmd = f"COPY {table_name}{table_columns} FROM STDIN WITH (FORMAT CSV, HEADER FALSE)"

        logger.debug("[Process {process_id}] Dumping CSV in buffer.")

        time_spent["other"] += current_time_millis() - session_setup_start

        csv_creation_start = current_time_millis()
        output = io.StringIO()
        curr_df.to_csv(output, sep=",", header=False, index=False, columns=["dataset_id", "file_id", "index", "label"])
        output.seek(0)
        time_spent["csv_creation"] += current_time_millis() - csv_creation_start

        db_insertion_start = current_time_millis()
        logger.debug(f"[Process {process_id}] Copying to DB.")
        cursor.copy_expert(cmd, output)
        conn.commit()

        time_spent["db_insertion"] += current_time_millis() - db_insertion_start

    @staticmethod
    def _fallback_copy_insertion(
        process_id: int, dataset_id: int, curr_df: pd.DataFrame, time_spent: dict, session: Session
    ) -> None:
        db_insertion_start = current_time_millis()
        session.bulk_insert_mappings(Sample, curr_df.to_dict("records"))
        session.commit()
        time_spent["db_insertion"] += current_time_millis() - db_insertion_start

    # pylint: disable=too-many-locals

    @staticmethod
    def _handle_file_paths(
        process_id: int,
        sample_dbinsertion_batchsize: int,
        file_paths: list[str],
        modyn_config: dict,
        data_file_extension: str,
        filesystem_wrapper: AbstractFileSystemWrapper,
        file_wrapper_type: str,
        timestamp: int,
        dataset_name: str,
        dataset_id: int,
        session: Optional[Session],  # When using multithreading, we cannot pass the session, hence it is Optional
    ) -> None:
        """Given a list of paths (in terms of a Modyn FileSystem) to files,
        check whether there are any new files and if so, add all samples from these files into the DB."""

        assert sample_dbinsertion_batchsize > 0, "Invalid sample_dbinsertion_batchsize"

        db_connection: Optional[StorageDatabaseConnection] = None

        if session is None:  # Multithreaded
            db_connection = StorageDatabaseConnection(modyn_config)
            db_connection.setup_connection()
            session = db_connection.session

        insertion_func = NewFileWatcher._fallback_copy_insertion
        if session.bind.dialect.name == "postgres":
            insertion_func = NewFileWatcher._postgres_copy_insertion

        dataset: Dataset = session.query(Dataset).filter(Dataset.name == dataset_name).first()

        def check_valid_file(file_path: str) -> bool:
            path_obj = pathlib.Path(file_path)
            if path_obj.suffix != data_file_extension:
                return False
            if (
                dataset.ignore_last_timestamp or filesystem_wrapper.get_modified(file_path) >= timestamp
            ) and NewFileWatcher._file_unknown(session, file_path):
                return True

            return False

        valid_files = list(filter(check_valid_file, file_paths))

        curr_df = pd.DataFrame(columns=["sample_id", "dataset_id", "file_id", "index", "label"])

        time_spent = {
            "init": 0,
            "file_creation": 0,
            "label_extraction": 0,
            "df_creation": 0,
            "csv_creation": 0,
            "db_insertion": 0,
            "other": 0,
        }

        for num_file, file_path in enumerate(valid_files):
            init_start = current_time_millis()

            file_wrapper = get_file_wrapper(
                file_wrapper_type, file_path, dataset.file_wrapper_config, filesystem_wrapper
            )
            number_of_samples = file_wrapper.get_number_of_samples()
            logger.debug(
                f"[Process {process_id}] Found new, unknown file: {file_path} with {number_of_samples} samples."
            )

            time_spent["init"] += current_time_millis() - init_start

            file_creation_start = current_time_millis()

            try:
                file: File = File(
                    dataset=dataset,
                    path=file_path,
                    created_at=filesystem_wrapper.get_created(file_path),
                    updated_at=filesystem_wrapper.get_modified(file_path),
                    number_of_samples=number_of_samples,
                )
                session.add(file)
                session.commit()
            except exc.SQLAlchemyError as exception:
                logger.critical(f"[Process {process_id}] Could not create file {file_path} in database: {exception}")
                session.rollback()
                continue

            file_id = file.file_id
            logger.info(
                f"[Process {process_id}] Extracting and inserting samples for file {file_path} (id = {file_id})"
            )

            time_spent["file_creation"] += current_time_millis() - file_creation_start
            label_extraction_start = current_time_millis()

            labels = file_wrapper.get_all_labels()
            logger.debug(
                f"[Process {process_id}] Labels extracted in"
                + f" {round((current_time_millis() - label_extraction_start) / 1000, 2)}s."
            )

            time_spent["label_extraction"] += current_time_millis() - label_extraction_start
            df_creation_start = current_time_millis()

            file_df = pd.DataFrame.from_dict({"dataset_id": dataset_id, "file_id": file_id, "label": labels})
            file_df["index"] = range(len(file_df))
            curr_df = pd.concat([curr_df, file_df])

            time_spent["df_creation"] += current_time_millis() - df_creation_start

            if len(curr_df) >= sample_dbinsertion_batchsize or num_file == len(valid_files) - 1:
                logger.debug(f"[Process {process_id}] Inserting {len(curr_df)} samples.")
                insertion_func_start = current_time_millis()
                insertion_func(process_id, dataset_id, curr_df, time_spent, session)

                logger.debug(
                    f"[Process {process_id}] Inserted {len(curr_df)} samples in"
                    + f" {round((current_time_millis() - insertion_func_start) / 1000, 2)}s."
                )

                cleanup_start = current_time_millis()

                curr_df = curr_df.iloc[0:0]
                time_spent["other"] += current_time_millis() - cleanup_start

        with open(f"/tmp/modyn_{current_time_millis()}_process{process_id}_stats.json", "w") as statsfile:
            json.dump(time_spent, statsfile)

        if db_connection is not None:
            db_connection.terminate_connection()

    def _update_files_in_directory(
        self,
        filesystem_wrapper: AbstractFileSystemWrapper,
        file_wrapper_type: str,
        path: str,
        timestamp: int,
        session: Session,
        dataset: Dataset,
    ) -> None:
        """Recursively get all files in a directory.

        Get all files that have a timestamp that is equal or greater than the given timestamp."""
        if not filesystem_wrapper.isdir(path):
            logger.critical(f"Path {path} is not a directory.")
            return

        data_file_extension = json.loads(dataset.file_wrapper_config)["file_extension"]
        file_paths = filesystem_wrapper.list(path, recursive=True)

        if self._disable_mt or (self._is_test and self._is_mac):
            NewFileWatcher._handle_file_paths(
                -1,
                self._sample_dbinsertion_batchsize,
                file_paths,
                self.modyn_config,
                data_file_extension,
                filesystem_wrapper,
                file_wrapper_type,
                timestamp,
                dataset.name,
                self.__dataset_id,
                session,
            )
            return

        process_start_time = current_time_millis()

        files_per_proc = int(len(file_paths) / self._insertion_threads)
        processes: list[mp.Process] = []
        for i in range(self._insertion_threads):
            start_idx = i * files_per_proc
            end_idx = start_idx + files_per_proc if i < self._insertion_threads - 1 else len(file_paths)
            paths = file_paths[start_idx:end_idx]

            if len(paths) > 0:
                proc = mp.Process(
                    target=NewFileWatcher._handle_file_paths,
                    args=(
                        i,
                        self._sample_dbinsertion_batchsize,
                        paths,
                        self.modyn_config,
                        data_file_extension,
                        filesystem_wrapper,
                        file_wrapper_type,
                        timestamp,
                        dataset.name,
                        self.__dataset_id,
                        None,
                    ),
                )
                proc.start()
                processes.append(proc)

        for proc in processes:
            proc.join()

        if len(processes) > 0:
            logger.debug(
                f"Processes finished running in in"
                + f" {round((current_time_millis() - process_start_time) / 1000, 2)}s."
            )

    def run(self) -> None:
        """Run the dataset watcher."""
        logger.info("Starting dataset watcher.")
        with StorageDatabaseConnection(self.modyn_config) as database:
            while not self.__should_stop.value:
                dataset = database.session.query(Dataset).filter(Dataset.dataset_id == self.__dataset_id).first()
                self._seek(database, dataset)
                time.sleep(dataset.file_watcher_interval)


def run_new_file_watcher(modyn_config: dict, dataset_id: int, should_stop: Any) -> None:
    """Run the file watcher for a dataset.

    Args:
        dataset_id (int): Dataset id.
        should_stop (Value): Value to check if the file watcher should stop.
    """
    file_watcher = NewFileWatcher(modyn_config, dataset_id, should_stop)
    file_watcher.run()
