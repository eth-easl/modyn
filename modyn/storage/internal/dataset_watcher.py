import typing
import logging
import time
import datetime
import uuid
from threading import Thread

from sqlalchemy.orm import sessionmaker
from sqlalchemy import exc

from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.storage_database_utils import get_filesystem_wrapper, get_file_wrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType


logger = logging.getLogger(__name__)


class DatasetWatcher:
    _session = None  # For testing purposes
    _testing = False  # For testing purposes

    def __init__(self, modyn_config: dict):
        self.modyn_config = modyn_config
        self._last_timestamp: datetime.datetime = datetime.datetime.min

    def _seek(self, timestamp: datetime.datetime) -> None:
        """
        Seek the filesystem for files with a timestamp that is equal or greater than the given timestamp.
        """
        logger.info(f'Seeking for files with a timestamp that is equal or greater than {timestamp}')
        session = self._get_database_session()
        datasets: typing.Optional[typing.List[Dataset]] = session.query(Dataset).all()

        if datasets is None or len(datasets) == 0:
            logger.warning('No datasets found.')
            return

        for dataset in datasets:
            filesystem_wrapper = self._get_filesystem_wrapper(dataset.filesystem_wrapper_type, dataset.base_path)

            if filesystem_wrapper.exists(dataset.base_path):
                print(f'Path {dataset.base_path} exists.')
                print(filesystem_wrapper.isdir(dataset.base_path))
                if filesystem_wrapper.isdir(dataset.base_path):
                    self._update_files_in_directory(filesystem_wrapper,
                                                    dataset.file_wrapper_type,
                                                    dataset.base_path,
                                                    timestamp,
                                                    session,
                                                    dataset)
                else:
                    logger.warning(f'Path {dataset.base_path} is not a directory.')
            else:
                logger.warning(f'Path {dataset.base_path} does not exist.')

    def _get_database_session(self) -> sessionmaker:
        if self._session is not None:
            return self._session
        with DatabaseConnection(self.modyn_config) as database:
            return database.get_session()

    def _get_filesystem_wrapper(self,
                                filesystem_wrapper_type: FilesystemWrapperType,
                                base_path: str) -> AbstractFileSystemWrapper:
        return get_filesystem_wrapper(filesystem_wrapper_type, base_path)

    def _get_file_wrapper(self, file_wrapper_type: FileWrapperType, file_path: str) -> AbstractFileWrapper:
        return get_file_wrapper(file_wrapper_type, file_path)

    def _update_files_in_directory(self,
                                   filesystem_wrapper: AbstractFileSystemWrapper,
                                   file_wrapper_type: str,
                                   path: str,
                                   timestamp: datetime.datetime,
                                   session: sessionmaker,
                                   dataset: Dataset) -> None:
        """
        Recursively get all files in a directory that have a timestamp that is equal or greater
        than the given timestamp.
        """
        if not filesystem_wrapper.isdir(path):
            logger.warning(f'Path {path} is not a directory.')
            raise ValueError(f'Path {path} is not a directory.')
        for file_path in filesystem_wrapper.list(path, recursive=True):
            if filesystem_wrapper.isfile(file_path):
                file_wrapper = self._get_file_wrapper(file_wrapper_type, file_path)
                if filesystem_wrapper.get_modified(file_path) >= timestamp and \
                        session.query(File).filter(File.path == file_path).first() is None:
                    try:
                        number_of_samples = file_wrapper.get_size()
                        file: File = File(
                            dataset=dataset,
                            path=file_path,
                            created_at=filesystem_wrapper.get_created(file_path),
                            updated_at=filesystem_wrapper.get_modified(file_path),
                            number_of_samples=number_of_samples
                        )
                        session.add(file)
                        session.commit()
                        for i in range(number_of_samples):
                            sample: Sample = Sample(
                                file=file,
                                external_key=str(uuid.uuid4()),
                                index=i
                            )
                            session.add(sample)
                        session.commit()
                    except exc.SQLAlchemyError as exception:
                        logger.warning(f'Could not create file {file_path}: {exception}')
                        session.rollback()
                        continue

    def run(self) -> None:
        """
        Run the dataset watcher.
        """
        logger.info('Starting dataset watcher.')
        threads = []
        try:
            while True:
                time.sleep(self.modyn_config['storage']['dataset_watcher']['interval'])
                thread = Thread(target=self._seek, args=(self._last_timestamp,))
                thread.start()
                threads.append(thread)
                self._last_timestamp = datetime.datetime.now()
                if self._testing:
                    break
        finally:
            logger.info('Stopping dataset_watcher.')
            for thread in threads:
                thread.join()
