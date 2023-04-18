from datetime import datetime

from modyn.model_storage.internal.database.model_storage_base import ModelStorageBase
from sqlalchemy import TIMESTAMP, Column, Integer, String


class TrainedModel(ModelStorageBase):
    """trained model from the trainer_server."""

    __tablename__ = "trained_models"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}
    model_id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String(120), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=False), default=datetime.now())

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Model {self.model_id}>"
