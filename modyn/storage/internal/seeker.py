import typing
import logging
import time
import datetime
import uuid
from threading import Thread

from sqlalchemy.orm import sessionmaker
from sqlalchemy import exc

from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import InvalidFileWrapperTypeException
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import InvalidFilesystemWrapperTypeException
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.storage_database_utils import get_filesystem_wrapper, get_file_wrapper


logger = logging.getLogger(__name__)


class Seeker:
    def __init__(self, modyn_config: dict):
        self.modyn_config = modyn_config
        self._last_timestamp: datetime.datetime = datetime.datetime.min

    def _seek(self, timestamp: datetime.datetime) -> None:
        """
        Seek the filesystem for files with a timestamp that is equal or greater than the given timestamp.
        """
        logger.info(f'Seeking for files with a timestamp that is equal or greater than {timestamp}')
        with DatabaseConnection(self.modyn_config) as database:
            session = database.get_session()
            datasets: typing.Optional[typing.List[Dataset]] = session.query(Dataset).all()

            if datasets is None:
                logger.warning('No datasets found.')
                return

            for dataset in datasets:
                try:
                    filesystem_wrapper = get_filesystem_wrapper(dataset.filesystem_wrapper_type, dataset.base_path)
                except InvalidFilesystemWrapperTypeException as exception:
                    logger.error(f'Error getting filesystem wrapper: {exception}')
                    continue

                if filesystem_wrapper.exists(dataset.base_path):
                    if filesystem_wrapper.isdir(dataset.base_path):
                        self._update_files_in_directory(filesystem_wrapper,
                                                        dataset.base_path,
                                                        timestamp,
                                                        session,
                                                        dataset)
                    else:
                        logger.warning(f'Path {dataset.base_path} is not a directory.')
                else:
                    logger.warning(f'Path {dataset.base_path} does not exist.')

    def _update_files_in_directory(self,
                                   filesystem_wrapper: AbstractFileSystemWrapper,
                                   path: str,
                                   timestamp: datetime.datetime,
                                   session: sessionmaker,
                                   dataset: Dataset) -> None:
        """
        Recursively get all files in a directory that have a timestamp that is equal or greater
        than the given timestamp.
        """
        for file_in_path in filesystem_wrapper.list(path):
            file_path = filesystem_wrapper.join(path, file_in_path)
            if filesystem_wrapper.isfile(file_path):
                try:
                    file_wrapper = get_file_wrapper(filesystem_wrapper.filesystem_wrapper_type, file_path)
                except InvalidFileWrapperTypeException as exception:
                    logger.warning(f'Could not get file wrapper for file {file_path}: {exception}')
                    continue
                if file_wrapper.get_modified() >= timestamp and \
                        session.query(File).filter(File.path == file_path).first() is None:
                    try:
                        file: File = File(
                            dataset=dataset,
                            path=file_path,
                            created_at=filesystem_wrapper.get_created(file_path),
                            updated_at=filesystem_wrapper.get_modified(file_path)
                        )
                        session.add(file)
                        for i in range(len(file_wrapper.get_size())):
                            sample: Sample = Sample(
                                file=file,
                                external_key=uuid.uuid4(),
                                index=i
                            )
                            session.add(sample)
                        session.commit()
                    except exc.SQLAlchemyError as exception:
                        logger.warning(f'Could not create file {file_path}: {exception}')
                        session.rollback()
                        continue
            elif filesystem_wrapper.isdir(file_path):
                self._update_files_in_directory(filesystem_wrapper, file_path, timestamp, session, dataset)

    def run(self) -> None:
        """
        Run the seeker.
        """
        logger.info('Starting seeker.')
        threads = []
        try:
            while True:
                time.sleep(self.modyn_config['storage']['seeker']['interval'])
                thread = Thread(target=self._seek, args=(self._last_timestamp,))
                thread.start()
                threads.append(thread)
                self._last_timestamp = datetime.datetime.now()
        except KeyboardInterrupt:
            logger.info('Stopping seeker.')
        finally:
            for thread in threads:
                thread.join()
