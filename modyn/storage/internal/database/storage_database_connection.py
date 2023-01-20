"""Database connection context manager."""

from __future__ import annotations

import logging

from modyn.database.abstract_database_connection import AbstractDatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.storage_base import Base
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType
from sqlalchemy import exc

logger = logging.getLogger(__name__)


class StorageDatabaseConnection(AbstractDatabaseConnection):
    """Database connection context manager."""

    def __init__(self, modyn_config: dict) -> None:
        """Initialize the database connection.

        Args:
            modyn_config (dict): Configuration of the modyn module.
        """
        super().__init__(modyn_config)
        self.drivername: str = self.modyn_config["storage"]["database"]["drivername"]
        self.username: str = self.modyn_config["storage"]["database"]["username"]
        self.password: str = self.modyn_config["storage"]["database"]["password"]
        self.host: str = self.modyn_config["storage"]["database"]["host"]
        self.port: int = self.modyn_config["storage"]["database"]["port"]
        self.database: str = self.modyn_config["storage"]["database"]["database"]

    def create_tables(self) -> None:
        """
        Create all tables. Each table is represented by a class.

        All classes that inherit from Base are mapped to tables
        which are created in the database if they do not exist.

        The metadata is a collection of Table objects that inherit from Base and their associated
        schema constructs (such as Column objects, ForeignKey objects, and so on).
        """
        Base.metadata.create_all(self.engine)

    def add_dataset(
        self,
        name: str,
        base_path: str,
        filesystem_wrapper_type: FilesystemWrapperType,
        file_wrapper_type: FileWrapperType,
        description: str,
        version: str,
        file_wrapper_config: str,
    ) -> bool:
        """
        Add dataset to database.

        If dataset with name already exists, it is updated.
        """
        try:
            if self.session.query(Dataset).filter(Dataset.name == name).first() is not None:
                logger.info(f"Dataset with name {name} exists.")
                self.session.query(Dataset).filter(Dataset.name == name).update(
                    {
                        "base_path": base_path,
                        "filesystem_wrapper_type": filesystem_wrapper_type,
                        "file_wrapper_type": file_wrapper_type,
                        "description": description,
                        "version": version,
                        "file_wrapper_config": file_wrapper_config,
                    }
                )
            else:
                dataset = Dataset(
                    name=name,
                    base_path=base_path,
                    filesystem_wrapper_type=filesystem_wrapper_type,
                    file_wrapper_type=file_wrapper_type,
                    description=description,
                    version=version,
                    file_wrapper_config=file_wrapper_config,
                )
                self.session.add(dataset)
            self.session.commit()
        except exc.SQLAlchemyError as exception:
            logger.error(f"Error adding dataset: {exception}")
            self.session.rollback()
            return False
        return True
