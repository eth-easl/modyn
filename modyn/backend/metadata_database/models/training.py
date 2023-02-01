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
