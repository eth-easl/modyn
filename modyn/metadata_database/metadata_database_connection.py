"""Database connection context manager."""

from __future__ import annotations

import logging

from modyn.database.abstract_database_connection import AbstractDatabaseConnection
from modyn.metadata_database.metadata_base import MetadataBase
from modyn.metadata_database.models import Pipeline
from modyn.metadata_database.models.selector_state_metadata import SelectorStateMetadata
from modyn.metadata_database.models.trained_models import TrainedModel
from sqlalchemy import func

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
        self.hash_partition_modulus: int = (
            self.modyn_config["metadata_database"]["hash_partition_modulus"]
            if "hash_partition_modulus" in self.modyn_config["metadata_database"]
            else 16
        )
        self.seed: int = (
            self.modyn_config["metadata_database"]["seed"] if "seed" in self.modyn_config["metadata_database"] else None
        )
        if self.seed is not None:
            if not -1 <= self.seed <= 1:
                raise ValueError("Postgres seed must be in [-1,1]")

    def __enter__(self) -> MetadataDatabaseConnection:
        """Create the engine and session. Then, if required, applies the seed.

        Returns:
            MetadataDatabaseConnection: MetadataDatabaseConnection.
        """
        super().__enter__()

        if self.seed is not None:
            self.session.execute(func.setseed(self.seed))

        return self

    def create_tables(self) -> None:
        """
        Create all tables. Each table is represented by a class.

        All classes that inherit from Base are mapped to tables
        which are created in the database if they do not exist.

        The metadata is a collection of Table objects that inherit from Base and their associated
        schema constructs (such as Column objects, ForeignKey objects, and so on).
        """
        MetadataBase.metadata.create_all(self.engine)

    def register_pipeline(self, num_workers: int, selection_strategy: str) -> int:
        """Register a new pipeline in the database.

        Args:
            num_workers (int): Number of workers in the pipeline.
            selection_strategy (str): The selection strategy to use

        Returns:
            int: Id of the newly created pipeline.
        """
        pipeline = Pipeline(num_workers=num_workers, selection_strategy=selection_strategy)
        self.session.add(pipeline)
        self.session.commit()
        pipeline_id = pipeline.pipeline_id
        return pipeline_id

    def add_selector_state_metadata_trigger(self, pipeline_id: int, trigger_id: int) -> None:
        """Add a new trigger to the selector state metadata table.

        This method creates a new partitions for the trigger.

        Args:
            pipeline_id (int): Id of the pipeline to which the trigger belongs.
            trigger_id (int): Id of the trigger.
        """
        SelectorStateMetadata.add_trigger(
            pipeline_id, trigger_id, self.session, self.engine, self.hash_partition_modulus
        )

    def add_trained_model(self, pipeline_id: int, trigger_id: int, model_path: str) -> int:
        """Add a trained model to the database.

        Args:
            pipeline_id: id of the pipeline it was created from.
            trigger_id: id of the trigger it was created.
            model_path: path on the local filesystem on which the model is stored.

        Returns:
            int: Id of the registered model
        """
        trained_model = TrainedModel(pipeline_id=pipeline_id, trigger_id=trigger_id, model_path=model_path)
        self.session.add(trained_model)
        self.session.commit()
        model_id = trained_model.model_id
        return model_id
