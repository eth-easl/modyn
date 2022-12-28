from sqlalchemy import create_engine
from sqlalchemy import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.file_system_wrapper.file_system_wrapper_type import FileSystemWrapperType
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType

Base = declarative_base()

logger = logging.getLogger(__name__)

class DatabaseConnection(): 
    """
    Database connection context manager.
    """

    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config

    def __enter__(self) -> DatabaseConnection:
        self.url = URL.create(
            drivername=self.modyn_config['database']['drivername'],
            username=self.modyn_config['database']['username'],
            password=self.modyn_config['database']['password'],
            host=self.modyn_config['database']['host'],
            port=self.modyn_config['database']['port'],
            database=self.modyn_config['database']['database']
        )
        self.engine = create_engine(self.url, echo=True)
        self.session = sessionmaker(bind=self.engine)()
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        self.session.close()
        self.engine.dispose()

    def get_session(self) -> sessionmaker:
        return self.session

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def add_dataset(self, name: str, base_path: str, file_system_wrapper_type: FileSystemWrapperType, file_wrapper_type: FileWrapperType, description: str) -> bool:
        try:
            if self.session.query(Dataset).filter(Dataset.name == name).first() is not None:
                logger.info('Dataset with name %s exists.', name)
            else:
                dataset = Dataset(name=name, base_path=base_path, file_system_wrapper_type=file_system_wrapper_type, file_wrapper_type=file_wrapper_type, description=description)
                self.session.add(dataset)
                self.session.commit()
        except Exception as e:
            logger.error('Error adding dataset: %s', e)
            self.session.rollback()
            return False
        return True
