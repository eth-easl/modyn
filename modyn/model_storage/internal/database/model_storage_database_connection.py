"""Database connection context manager."""

from __future__ import annotations

import logging

from modyn.database.abstract_database_connection import AbstractDatabaseConnection
from modyn.model_storage.internal.database.model_storage_base import ModelStorageBase

logger = logging.getLogger(__name__)


class ModelStorageDatabaseConnection(AbstractDatabaseConnection):
    """Database connection context manager."""

    def __init__(self, modyn_config: dict) -> None:
        """Initialize the database connection.

        Args:
            modyn_config (dict): Configuration of the modyn module.
        """
        super().__init__(modyn_config)
        self.drivername: str = self.modyn_config["model_storage"]["database"]["drivername"]
        self.username: str = self.modyn_config["model_storage"]["database"]["username"]
        self.password: str = self.modyn_config["model_storage"]["database"]["password"]
        self.host: str = self.modyn_config["model_storage"]["database"]["host"]
        self.port: int = self.modyn_config["model_storage"]["database"]["port"]
        self.database: str = self.modyn_config["model_storage"]["database"]["database"]

    def create_tables(self) -> None:
        """
        Create all tables. Each table is represented by a class.

        All classes that inherit from Base are mapped to tables
        which are created in the database if they do not exist.

        The metadata is a collection of Table objects that inherit from Base and their associated
        schema constructs (such as Column objects, ForeignKey objects, and so on).
        """
        ModelStorageBase.metadata.create_all(self.engine)
