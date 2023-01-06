import logging
import time
import uuid
from typing import Optional

from sqlalchemy.orm import sessionmaker
from sqlalchemy import exc
from sqlalchemy.orm.session import Session

from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.storage_database_utils import get_filesystem_wrapper, get_file_wrapper


logger = logging.getLogger(__name__)


class NewFileWatcher:

    def __init__(self, modyn_config: dict):
        self.modyn_config = modyn_config
        self._last_timestamp: int = 0

    def _seek(self, timestamp: int) -> None:
        """
        Seek the filesystem for files with a timestamp that is equal or greater than the given timestamp.
        """
        logger.debug(f'Seeking for files with a timestamp that is equal or greater than {timestamp}')
        with DatabaseConnection(self.modyn_config) as database:
            session = database.get_session()

            datasets = self._get_datasets(session)

            for dataset in datasets:
                self._seek_dataset(session, dataset, timestamp)

    def _seek_dataset(self, session: Session, dataset: Dataset, timestamp: int) -> None:
        filesystem_wrapper = get_filesystem_wrapper(dataset.filesystem_wrapper_type, dataset.base_path)

        if filesystem_wrapper.exists(dataset.base_path):
            if filesystem_wrapper.isdir(dataset.base_path):
                print(f'Path {dataset.base_path} is a directory.')
                self._update_files_in_directory(filesystem_wrapper,
                                                dataset.file_wrapper_type,
                                                dataset.base_path,
                                                timestamp,
                                                session,
                                                dataset)
            else:
                logger.critical(f'Path {dataset.base_path} is not a directory.')
        else:
            logger.warning(f'Path {dataset.base_path} does not exist.')

    def _get_datasets(self, session: Session) -> list[Dataset]:
        datasets: Optional[list[Dataset]] = session.query(Dataset).all()

        if datasets is None or len(datasets) == 0:
            logger.warning('No datasets found.')
            return []

        return datasets

    def _file_unknown(self, session: Session, file_path: str) -> bool:
        return session.query(File).filter(File.path == file_path).first() is None

    def _update_files_in_directory(self,
                                   filesystem_wrapper: AbstractFileSystemWrapper,
                                   file_wrapper_type: str,
                                   path: str,
                                   timestamp: int,
                                   session: sessionmaker,
                                   dataset: Dataset) -> None:
        """
        Recursively get all files in a directory that have a timestamp that is equal or greater
        than the given timestamp.
        """
        if not filesystem_wrapper.isdir(path):
            logger.critical(f'Path {path} is not a directory.')
            return
        for file_path in filesystem_wrapper.list(path, recursive=True):
            file_wrapper = get_file_wrapper(file_wrapper_type, file_path, dataset.file_wrapper_config)
            if filesystem_wrapper.get_modified(file_path) >= timestamp and self._file_unknown(session, file_path):
                try:
                    number_of_samples = file_wrapper.get_number_of_samples()
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
                    logger.warning(f'Could not create file {file_path} in database: {exception}')
                    session.rollback()
                    continue

    def run(self) -> None:
        """
        Run the dataset watcher.
        """
        logger.info('Starting dataset watcher.')
        while self._last_timestamp >= 0:
            time.sleep(self.modyn_config['storage']['new_file_watcher']['interval'])
            self._seek(self._last_timestamp)
            self._last_timestamp = int(time.time() * 1000)
