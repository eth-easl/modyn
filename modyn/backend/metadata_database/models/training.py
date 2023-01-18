"""Training model."""

from modyn.backend.metadata_database.metadata_base import Base
from sqlalchemy import Column, Integer


class Training(Base):
    """Training model."""

    __tablename__ = "trainings"
    id = Column(Integer, primary_key=True)
    number_of_workers = Column(Integer, nullable=False)
    training_set_size = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Training {self.id}>"

    def __init__(self, number_of_workers: int, training_set_size: int):
        """Init training.

        Args:
            number_of_workers (int): number of workers
            training_set_size (int): training set size
        """
        self.number_of_workers = number_of_workers
        self.training_set_size = training_set_size
