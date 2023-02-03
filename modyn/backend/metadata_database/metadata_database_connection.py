"""Database connection context manager."""

from __future__ import annotations

import logging

from modyn.backend.metadata_database.metadata_base import MetadataBase
from modyn.backend.metadata_database.models.pipeline import Pipeline
from modyn.database.abstract_database_connection import AbstractDatabaseConnection

logger = logging.getLogger(__name__)


class MetadataDatabaseConnection(AbstractDatabaseConnection):
    """Database connection context manager."""

    def __init__(self, modyn_config: dict) -> None:
        """Initialize the database connection.

        Args:
            modyn_config (dict): Configuration of the modyn module.
        """
        super().__init__(modyn_config)
        self.drivername: str = self.modyn_config["metadata_database"]["drivername"]
        self.username: str = self.modyn_config["metadata_database"]["username"]
        self.password: str = self.modyn_config["metadata_database"]["password"]
        self.host: str = self.modyn_config["metadata_database"]["host"]
        self.port: int = self.modyn_config["metadata_database"]["port"]
        self.database: str = self.modyn_config["metadata_database"]["database"]

    def create_tables(self) -> None:
        """
        Create all tables. Each table is represented by a class.

        All classes that inherit from Base are mapped to tables
        which are created in the database if they do not exist.

        The metadata is a collection of Table objects that inherit from Base and their associated
        schema constructs (such as Column objects, ForeignKey objects, and so on).
        """
        MetadataBase.metadata.create_all(self.engine)

    def register_pipeline(self, num_workers: int) -> int:
        """Register a new pipeline in the database.

        Args:
            num_workers (int): Number of workers in the pipeline.

        Returns:
            int: Id of the newly created pipeline.
        """
        pipeline = Pipeline(num_workers=num_workers)
        self.session.add(pipeline)
        self.session.commit()
        pipeline_id = pipeline.pipeline_id
        return pipeline_id
