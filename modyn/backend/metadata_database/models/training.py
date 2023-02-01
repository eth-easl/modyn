"""Training model."""

from modyn.backend.metadata_database.metadata_base import Base
from sqlalchemy import Column, Integer


class Training(Base):
    """Training model."""

    __tablename__ = "trainings"
    training_id = Column("training_id", Integer, primary_key=True)
    number_of_workers = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Training {self.training_id}>"

    def __init__(self, number_of_workers: int):
        """Init training.

        Args:
            number_of_workers (int): number of workers
            training_set_size (int): training set size
        """
        self.number_of_workers = number_of_workers
