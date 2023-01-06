"""Database connection context manager."""

from __future__ import annotations
import logging

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import Session
from sqlalchemy import create_engine, exc
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.database.base import Base

logger = logging.getLogger(__name__)


class DatabaseConnection():
    """Database connection context manager."""

    session: Session = None
    engine: Engine = None
    url = None

    def __init__(self, modyn_config: dict) -> None:
        """Initialize the database connection.

        Args:
            modyn_config (dict): Configuration of the modyn module.
        """
        self.modyn_config = modyn_config

    def __enter__(self) -> DatabaseConnection:
        """Create the engine and session.

        Returns:
            DatabaseConnection: DatabaseConnection.
        """
        self.url = URL.create(
            drivername=self.modyn_config['storage']['database']['drivername'],
            username=self.modyn_config['storage']['database']['username'],
            password=self.modyn_config['storage']['database']['password'],
            host=self.modyn_config['storage']['database']['host'],
            port=self.modyn_config['storage']['database']['port'],
            database=self.modyn_config['storage']['database']['database']
        )
        self.engine = create_engine(self.url, echo=True)
        self.session = sessionmaker(bind=self.engine)()
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        """Close the session and dispose the engine.

        Args:
            exc_type (type): exception type
            exc_val (Exception): exception value
            exc_tb (Exception): exception traceback
        """
        self.session.close()
        self.engine.dispose()

    def get_session(self) -> Session:
        """Get the session.

        Returns:
            Session: Session.
        """
        return self.session

    def create_all(self) -> None:
        """
        Create all tables. Each table is represented by a class.

        All classes that inherit from Base are mapped to tables
        which are created in the database if they do not exist.

        The metadata is a collection of Table objects that inherit from Base and their associated
        schema constructs (such as Column objects, ForeignKey objects, and so on).
        """
        Base.metadata.create_all(self.engine)

    def add_dataset(self, name: str, base_path: str,
                    filesystem_wrapper_type: FilesystemWrapperType,
                    file_wrapper_type: FileWrapperType, description: str, version: str,
                    file_wrapper_config: str) -> bool:
        """
        Add dataset to database.

        If dataset with name already exists, it is updated.
        """
        try:
            if self.session.query(Dataset).filter(Dataset.name == name).first() is not None:
                logger.info(f'Dataset with name {name} exists.')
                self.session.query(Dataset).filter(Dataset.name == name).update({
                    'base_path': base_path,
                    'filesystem_wrapper_type': filesystem_wrapper_type,
                    'file_wrapper_type': file_wrapper_type,
                    'description': description,
                    'version': version,
                    'file_wrapper_config': file_wrapper_config
                })
            else:
                dataset = Dataset(name=name,
                                  base_path=base_path,
                                  filesystem_wrapper_type=filesystem_wrapper_type,
                                  file_wrapper_type=file_wrapper_type,
                                  description=description,
                                  version=version,
                                  file_wrapper_config=file_wrapper_config)
                self.session.add(dataset)
            self.session.commit()
        except exc.SQLAlchemyError as exception:
            logger.error(f'Error adding dataset: {exception}')
            self.session.rollback()
            return False
        return True
