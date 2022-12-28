import typing
import logging 
import time
import datetime
import uuid 
from threading import Thread

from sqlalchemy.orm import sessionmaker

from modyn.storage.internal.file_system_wrapper.abstract_file_system_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.file_system_wrapper.file_system_wrapper_type import FileSystemWrapperType
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.utils import get_file_system_wrapper, get_file_wrapper

from utils import dynamic_module_import

logger = logging.getLogger(__name__)

class Seeker:
    def __init__(self, modyn_config: dict):
        self.modyn_config = modyn_config
        self._last_timestamp: datetime.datetime = datetime.datetime.min

    def _seek(self, timestamp: datetime.datetime) -> typing.List[str]:
        """
        Seek the filesystem for files with a timestamp that is equal or greater than the given timestamp.
        """
        logger.info('Seeking for files with a timestamp that is equal or greater than %s', timestamp)
        with DatabaseConnection(self.modyn_config) as database:
            session = database.get_session()
            datasets: typing.Optional[typing.List[Dataset]] = session.query(Dataset).all()

            for dataset in datasets:
                file_system_wrapper = get_file_system_wrapper(dataset.file_system_wrapper_type, dataset.base_path)
                if file_system_wrapper.exists(dataset.base_path):
                    if file_system_wrapper.isdir(dataset.base_path):
                        self._update_files_in_directory(file_system_wrapper, dataset.base_path, timestamp, session, dataset)
                    else:
                        logger.warning('Path %s is not a directory.', dataset.base_path)
                else:
                    logger.warning('Path %s does not exist.', dataset.base_path)

    def _update_files_in_directory(self, file_system_wrapper: AbstractFileSystemWrapper, path: str, timestamp: datetime.datetime, session: sessionmaker, dataset: Dataset) -> None:
        """
        Recursively get all files in a directory that have a timestamp that is equal or greater than the given timestamp.
        """
        for file in file_system_wrapper.list(path):
            file_path = file_system_wrapper.join(path, file)
            if file_system_wrapper.isfile(file_path):
                try:
                    file_wrapper = get_file_wrapper(file_system_wrapper.filesystem_wrapper_type, file_path)
                except Exception as e:
                    logger.warning('Could not get file wrapper for file %s: %s', file_path, e)
                    continue
                if file_wrapper.get_modified() >= timestamp and session.query(File).filter(File.path == file_path).first() is None:
                    try:
                        file: File = File(
                            dataset=dataset,
                            path=file_path,
                            created=file_wrapper.get_created(),
                            modified=file_wrapper.get_modified()
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
                    except Exception as e:
                        logger.warning('Could not create file %s: %s', file_path, e)
                        session.rollback()
                        continue
            elif file_system_wrapper.isdir(file_path):
                self._get_files_in_directory_with_timestamp(file_system_wrapper, file_path, timestamp)

    def run(self):
        """
        Run the seeker.
        """
        threads = []
        try:
            while(True):
                time.sleep(self.modyn_config['storage']['seeker']['interval'])
                threads.append(Thread(target=self._seek, args=(self._last_timestamp,)).start())
                self._last_timestamp = datetime.datetime.now()
        except KeyboardInterrupt:
            logger.info('Stopping seeker.')
        except Exception as e:
            logger.error('Error in seeker: %s', e)
        finally:
            for thread in threads:
                thread.join()