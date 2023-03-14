"""New file watcher."""

import json
import logging
import pathlib
import time
from typing import Any, Optional

from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.database.storage_database_utils import get_file_wrapper, get_filesystem_wrapper
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
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

    def _file_unknown(self, session: Session, file_path: str) -> bool:
        """Check if a file is unknown.

        TODO (#147): This is a very inefficient way to check if a file is unknown. It should be replaced
        by a more efficient method.
        """
        return session.query(File).filter(File.path == file_path).first() is None

    # pylint: disable=too-many-locals

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
        for file_path in filesystem_wrapper.list(path, recursive=True):
            if pathlib.Path(file_path).suffix != data_file_extension:
                continue
            if (
                dataset.ignore_last_timestamp or filesystem_wrapper.get_modified(file_path) >= timestamp
            ) and self._file_unknown(session, file_path):
                file_wrapper = get_file_wrapper(
                    file_wrapper_type, file_path, dataset.file_wrapper_config, filesystem_wrapper
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
                    continue

                file_id = file.file_id
                logger.debug(f"Encountered new file and inserted with file id = {file_id}: {file_path}")
                logger.info(f"Extracting and inserting samples for file {file_path}")
                labels = file_wrapper.get_all_labels()

                try:
                    samples = [
                        Sample(file=file, file_id=file_id, index=i, label=labels[i]) for i in range(number_of_samples)
                    ]
                    logger.debug("Samples generated, inserting.")
                    session.bulk_save_objects(samples)
                    session.commit()
                    logger.debug(f"Inserted {number_of_samples} samples.")
                except exc.SQLAlchemyError as exception:
                    logger.warning(f"Could not create samples for file {file_path} in database: {exception}")
                    session.rollback()
                    session.delete(file)
                    continue

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
