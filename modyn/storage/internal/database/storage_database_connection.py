"""Database connection context manager."""

from __future__ import annotations

import logging

from modyn.database.abstract_database_connection import AbstractDatabaseConnection
from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.database.storage_base import StorageBase
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
        self.hash_partition_modulus: int = (
            self.modyn_config["storage"]["database"]["hash_partition_modulus"]
            if "hash_partition_modulus" in self.modyn_config["storage"]["database"]
            else 8
        )
        self.sample_table_unlogged: bool = (
            self.modyn_config["storage"]["database"]["sample_table_unlogged"]
            if "sample_table_unlogged" in self.modyn_config["storage"]["database"]
            else True
        )

    def __enter__(self) -> StorageDatabaseConnection:
        """Create the engine and session.

        Returns:
            DatabaseConnection: DatabaseConnection.
        """
        super().__enter__()
        return self

    def create_tables(self) -> None:
        """
        Create all tables. Each table is represented by a class.

        All classes that inherit from Base are mapped to tables
        which are created in the database if they do not exist.

        The metadata is a collection of Table objects that inherit from Base and their associated
        schema constructs (such as Column objects, ForeignKey objects, and so on).
        """
        Sample.ensure_pks_correct(self.session)
        StorageBase.metadata.create_all(self.engine)

    def add_dataset(
        self,
        name: str,
        base_path: str,
        filesystem_wrapper_type: FilesystemWrapperType,
        file_wrapper_type: FileWrapperType,
        description: str,
        version: str,
        file_wrapper_config: str,
        ignore_last_timestamp: bool = False,
        file_watcher_interval: int = 5,
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
                        "ignore_last_timestamp": ignore_last_timestamp,
                        "file_watcher_interval": file_watcher_interval,
                    }
                )
            else:
                logger.info(f"Dataset with name {name} does not exist.")
                dataset = Dataset(
                    name=name,
                    base_path=base_path,
                    filesystem_wrapper_type=filesystem_wrapper_type,
                    file_wrapper_type=file_wrapper_type,
                    description=description,
                    version=version,
                    file_wrapper_config=file_wrapper_config,
                    last_timestamp=-1,  # Set to -1 as this is a new dataset
                    ignore_last_timestamp=ignore_last_timestamp,
                    file_watcher_interval=file_watcher_interval,
                )
                self.session.add(dataset)
            self.session.commit()
        except exc.SQLAlchemyError as exception:
            logger.error(f"Error adding dataset: {exception}")
            self.session.rollback()
            return False
        return True

    def delete_dataset(self, name: str) -> bool:
        """Delete dataset from database."""
        try:
            self.session.query(Sample).filter(
                Sample.file_id.in_(self.session.query(File.file_id).join(Dataset).filter(Dataset.name == name))
            ).delete(synchronize_session="fetch")
            self.session.query(File).filter(
                File.dataset_id.in_(self.session.query(Dataset.dataset_id).filter(Dataset.name == name))
            ).delete(synchronize_session="fetch")
            self.session.query(Dataset).filter(Dataset.name == name).delete(synchronize_session="fetch")
            self.session.commit()
        except exc.SQLAlchemyError as exception:
            logger.error(f"Error deleting dataset: {exception}")
            self.session.rollback()
            return False
        return True

    def add_sample_dataset(self, dataset_id: int) -> None:
        """Add a new dataset to the samples table.

        This method creates a new partitions for the dataset.

        Args:
            dataset_id (int): Id of the dataset
        """
        Sample.add_dataset(
            dataset_id, self.session, self.engine, self.hash_partition_modulus, self.sample_table_unlogged
        )
