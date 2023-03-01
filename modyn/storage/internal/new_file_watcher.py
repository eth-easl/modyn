"""New file watcher."""

import json
import logging
import multiprocessing
import os
import pathlib
import time
import uuid
from typing import Any, Optional, Type

from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.database.storage_database_utils import get_file_wrapper, get_filesystem_wrapper
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from sqlalchemy import exc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

logger = logging.getLogger(__name__)


class NewFileWatcher:
    """New file watcher.

    This class is responsible for watching all the filesystems of the datasets for new files. If a new file is found, it
    will be added to the database.
    """

    def __init__(self, modyn_config: dict, should_stop: Any):  # See https://github.com/python/typeshed/issues/8799
        """Initialize the new file watcher.

        Args:
            modyn_config (dict): Configuration of the modyn module.
            should_stop (Any): Value that indicates if the new file watcher should stop.
        """
        self.modyn_config = modyn_config
        self.__should_stop = should_stop

    def _seek(self) -> None:
        """Seek the filesystem for all the datasets for new files and add them to the database.

        If last timestamp is not ignored, the last timestamp of the dataset will be used to only
        seek for files that have a timestamp that is equal or greater than the last timestamp.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            session = database.session

            datasets = self._get_datasets(session)

            for dataset in datasets:
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

    # pylint: disable=too-many-locals
    @staticmethod
    def _handle_file_paths(
        file_paths: list[pathlib.Path],
        modyn_config: dict,
        data_file_extension: str,
        filesystem_wrapper: AbstractFileSystemWrapper,
        file_wrapper_type: str,
        timestamp: int,
        dataset: Dataset,
        forced_file_wrapper: Optional[Type],
    ) -> None:
        with StorageDatabaseConnection(modyn_config) as database:
            session = database.session
            for file_path in file_paths:
                if pathlib.Path(file_path).suffix != data_file_extension:
                    return

                if (
                    dataset.ignore_last_timestamp or filesystem_wrapper.get_modified(file_path) >= timestamp
                ) and NewFileWatcher._file_unknown(session, str(file_path)):
                    file_wrapper = get_file_wrapper(
                        file_wrapper_type,
                        file_path,
                        dataset.file_wrapper_config,
                        filesystem_wrapper,
                        forced_file_wrapper,
                    )

                    try:
                        number_of_samples = file_wrapper.get_number_of_samples()
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
                        logger.warning(f"Could not create file {file_path} in database: {exception}")
                        session.rollback()
                        return

                    file_id = file.file_id
                    logger.debug(f"Encountered new file and inserted with file id = {file_id}: {file_path}")
                    logger.info(f"Extracting and inserting samples for file {file_path}")
                    labels = file_wrapper.get_all_labels()
                    logger.info(f"Extracted labels for file {file_path}")

                    try:
                        samples = [
                            Sample(file=file, file_id=file_id, external_key=str(uuid.uuid4()), index=i, label=labels[i])
                            for i in range(number_of_samples)
                        ]
                        logger.debug("Samples generated, inserting.")
                        session.bulk_save_objects(samples)
                        session.commit()
                        logger.info(f"Inserted {number_of_samples} samples.")
                    except exc.SQLAlchemyError as exception:
                        logger.error(f"Could not create samples for file {file_path} in database: {exception}")
                        session.rollback()
                        session.delete(file)
                        return

    # TODO(MaxiBoether): fix unused arg, explain forced_file_wrapper
    # TODO(MaxiBoether): this function is currently only tested together with handle paths
    # TODO(MaxiBoether): hence we should write separate test functions
    # pylint: disable=too-many-locals, unused-argument
    def _update_files_in_directory(
        self,
        filesystem_wrapper: AbstractFileSystemWrapper,
        file_wrapper_type: str,
        path: str,
        timestamp: int,
        session: sessionmaker,
        dataset: Dataset,
        forced_file_wrapper: Optional[Type] = None,
    ) -> None:
        """Recursively get all files in a directory.

        Get all files that have a timestamp that is equal or greater than the given timestamp."""
        if not filesystem_wrapper.isdir(path):
            logger.critical(f"Path {path} is not a directory.")
            return

        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError as error:
            if multiprocessing.get_start_method() != "spawn" and "PYTEST_CURRENT_TEST" not in os.environ:
                logger.error("Start method is already set to {}", multiprocessing.get_start_method())
                raise error

        data_file_extension = json.loads(dataset.file_wrapper_config)["file_extension"]

        file_paths = filesystem_wrapper.list(path, recursive=True)

        num_procs = 23  # TODO(MaxiBoether): add config
        files_per_proc = int(len(file_paths) / num_procs)
        processes: list[multiprocessing.Process] = []

        for i in range(num_procs):
            start_idx = i * files_per_proc
            end_idx = start_idx + files_per_proc if i < num_procs - 1 else len(file_paths)
            paths = file_paths[start_idx:end_idx]
            if len(paths) > 0:
                logger.error("starting porc")
                proc = multiprocessing.Process(
                    target=NewFileWatcher._handle_file_paths,
                    args=(
                        paths,
                        self.modyn_config,
                        data_file_extension,
                        filesystem_wrapper,
                        file_wrapper_type,
                        timestamp,
                        dataset,
                        forced_file_wrapper,
                    ),
                )
                proc.start()
                processes.append(proc)

        for proc in processes:
            proc.join()

    def run(self) -> None:
        """Run the dataset watcher."""
        logger.info("Starting dataset watcher.")
        while not self.__should_stop.value:  # type: ignore  # See https://github.com/python/typeshed/issues/8799  # noqa: E501
            time.sleep(self.modyn_config["storage"]["new_file_watcher"]["interval"])
            self._seek()


def run_watcher(modyn_config: dict, should_stop: Any) -> None:  # See https://github.com/python/typeshed/issues/8799
    """Run the new file watcher.

    Args:
        modyn_config (dict): Configuration of the modyn module.
        should_stop (Value): Value that indicates if the watcher should stop.
    """
    watcher = NewFileWatcher(modyn_config, should_stop)
    watcher.run()
