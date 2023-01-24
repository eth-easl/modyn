"""Database connection context manager."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from modyn.backend.metadata_database.metadata_base import Base
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.metadata_database.models.training import Training
from modyn.database.abstract_database_connection import AbstractDatabaseConnection
from sqlalchemy import exc

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
        Base.metadata.create_all(self.engine)

    def set_metadata(
        self,
        keys: list[str],
        scores: list[float],
        seens: list[bool],
        labels: list[int],
        datas: list[bytes],
        training_id: int,
    ) -> None:
        for i, key in enumerate(keys):
            try:
                if self.session.query(Metadata).filter(Metadata.key == key).first() is not None:
                    self.session.query(Metadata).filter(Metadata.key == key).delete()
                metadata = Metadata(
                    key=key, score=scores[i], seen=seens[i], label=labels[i], data=datas[i], training_id=training_id
                )
                self.session.add(metadata)
                self.session.commit()
            except exc.SQLAlchemyError as exception:
                logger.error(f"Could not set metadata: {exception}")
                self.session.rollback()
                continue

    def delete_training(self, training_id: int) -> None:
        """Delete training.

        Args:
            training_id (int): training id
        """
        try:
            self.session.query(Metadata).filter(Metadata.training_id == training_id).delete()
            self.session.query(Training).filter(Training.training_id == training_id).delete()
            self.session.commit()
        except exc.SQLAlchemyError as exception:
            logger.error(f"Could not delete training: {exception}")
            self.session.rollback()

    def register_training(self, number_of_workers: int, training_set_size: int) -> Optional[int]:
        """Register training.

        Args:
            number_of_workers (int): number of workers
            training_set_size (int): training set size

        Returns:
            int: training id
        """
        try:
            training = Training(number_of_workers=number_of_workers, training_set_size=training_set_size)
            self.session.add(training)
            self.session.commit()
            return training.training_id
        except exc.SQLAlchemyError as exception:
            logger.error(f"Could not register training: {exception}")
            self.session.rollback()
            return None

    def get_training_information(self, training_id: int) -> Tuple[Optional[int], Optional[int]]:
        """Get training.

        Args:
            training_id (int): training id

        Returns:
            Tuple[int, int]: number of workers, training set size
        """
        try:
            training = self.session.query(Training).filter(Training.training_id == training_id).first()
            if training is None:
                return None, None
            return training.number_of_workers, training.training_set_size
        except exc.SQLAlchemyError as exception:
            logger.error(f"Could not get training: {exception}")
            self.session.rollback()
            return None, None
