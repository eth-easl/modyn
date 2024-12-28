"""Database connection context manager."""

from __future__ import annotations

import logging

from sqlalchemy import func

from modyn.database.abstract_database_connection import AbstractDatabaseConnection
from modyn.metadata_database.metadata_base import MetadataBase
from modyn.metadata_database.models import Pipeline
from modyn.metadata_database.models.selector_state_metadata import SelectorStateMetadata
from modyn.metadata_database.models.trained_models import TrainedModel
from modyn.metadata_database.utils import ModelStorageStrategyConfig

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
        self.host: str = self.modyn_config["metadata_database"]["hostname"]
        self.port: int = self.modyn_config["metadata_database"]["port"]
        self.database: str = self.modyn_config["metadata_database"]["database"]
        self.hash_partition_modulus: int = (
            self.modyn_config["metadata_database"]["hash_partition_modulus"]
            if "hash_partition_modulus" in self.modyn_config["metadata_database"]
            else 16
        )
        self.seed: int | None = (
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
        """Create all tables. Each table is represented by a class.

        All classes that inherit from Base are mapped to tables which
        are created in the database if they do not exist.

        The metadata is a collection of Table objects that inherit from
        Base and their associated schema constructs (such as Column
        objects, ForeignKey objects, and so on).
        """
        MetadataBase.metadata.create_all(self.engine)

    def register_pipeline(
        self,
        num_workers: int,
        model_class_name: str,
        model_config: str,
        amp: bool,
        selection_strategy: str,
        data_config: str,
        full_model_strategy: ModelStorageStrategyConfig,
        incremental_model_strategy: ModelStorageStrategyConfig | None = None,
        full_model_interval: int | None = None,
        auxiliary_pipeline_id: int | None = None,
    ) -> int:
        """Register a new pipeline in the database.

        Args:
            num_workers (int): Number of workers in the pipeline.
            model_class_name (str): the model class name that is used by the pipeline.
            model_config (str): the serialized model configuration options.
            amp (bool): whether amp is enabled for the model.
            selection_strategy (str): The selection strategy to use
            full_model_strategy: the strategy used to store full models.
            data_config: The configuration of the training dataset.
            incremental_model_strategy: the (optional) strategy used to store models incrementally.
            full_model_interval: the (optional) interval between which the full model strategy is used. If not set,
                                 the first model is stored according to the full model strategy, and the remaining
                                 by using the incremental model strategy.
            auxiliary_pipeline_id: (optional) the id of the auxiliary pipeline, used to store extra information. For
            example, for RHO-LOSS downsampling strategy it is used to store the IL models and holdout set.
        Returns:
            int: Id of the newly created pipeline.
        """
        pipeline = Pipeline(
            num_workers=num_workers,
            model_class_name=model_class_name,
            model_config=model_config,
            amp=amp,
            selection_strategy=selection_strategy,
            data_config=data_config,
            auxiliary_pipeline_id=auxiliary_pipeline_id,
            full_model_strategy_name=full_model_strategy.name,
            full_model_strategy_zip=full_model_strategy.zip,
            full_model_strategy_zip_algorithm=full_model_strategy.zip_algorithm,
            full_model_strategy_config=full_model_strategy.config,
        )
        if incremental_model_strategy:
            pipeline.inc_model_strategy_name = incremental_model_strategy.name
            pipeline.inc_model_strategy_zip = incremental_model_strategy.zip
            pipeline.inc_model_strategy_zip_algorithm = incremental_model_strategy.zip_algorithm
            pipeline.inc_model_strategy_config = incremental_model_strategy.config
        pipeline.full_model_interval = full_model_interval
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

    def add_trained_model(
        self,
        pipeline_id: int,
        trigger_id: int,
        model_path: str,
        metadata_path: str,
        parent_model: int | None = None,
    ) -> int:
        """Add a trained model to the database. Whenever the parent model is
        not specified, the model is expected to be fully stored, i.e., by
        applying a full model strategy.

        Args:
            pipeline_id: id of the pipeline it was created from.
            trigger_id: id of the trigger it was created.
            model_path: path on the local filesystem on which the model is stored.
            metadata_path: the path on the local filesystem where model metadata is stored.
            parent_model: (optional) id of the parent model.
        Returns:
            int: Id of the registered model
        """
        trained_model = TrainedModel(
            pipeline_id=pipeline_id,
            trigger_id=trigger_id,
            model_path=model_path,
            metadata_path=metadata_path,
            parent_model=parent_model,
        )
        self.session.add(trained_model)
        self.session.commit()
        model_id = trained_model.model_id
        return model_id

    def get_model_configuration(self, pipeline_id: int) -> tuple[str, str, bool]:
        """Get the model id and its configuration options for a given pipeline.

        Args:
            pipeline_id: id of the pipeline from which we want to extract the model.

        Returns:
            (str, str, bool): the model class name, its configuration options and if amp is enabled.
        """
        pipeline: Pipeline = self.session.query(Pipeline).get(pipeline_id)
        return pipeline.model_class_name, pipeline.model_config, pipeline.amp
