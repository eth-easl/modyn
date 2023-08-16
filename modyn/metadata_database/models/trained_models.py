"""Trained models downloaded from trainer server"""
from datetime import datetime

from modyn.metadata_database.metadata_base import MetadataBase
from modyn.metadata_database.models.triggers import Trigger
from sqlalchemy import TIMESTAMP, Column, ForeignKey, ForeignKeyConstraint, Integer, String
from sqlalchemy.orm import relationship


class TrainedModel(MetadataBase):
    """Trained model."""

    __tablename__ = "trained_models"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    model_id = Column("model_id", Integer, primary_key=True, autoincrement=True)
    pipeline_id = Column("pipeline_id", Integer)
    trigger_id = Column("trigger_id", Integer)
    timestamp = Column("timestamp", TIMESTAMP(timezone=False), default=datetime.now())
    model_path = Column("model_path", String(length=200), nullable=False)
    metadata_path = Column("metadata_path", String(length=200), nullable=True, default=None)
    parent_model = Column("parent_model", Integer, ForeignKey(f"{__tablename__}.model_id"), nullable=True, default=None)
    children = relationship("TrainedModel")
    __table_args__ = (
        ForeignKeyConstraint([pipeline_id, trigger_id], [Trigger.pipeline_id, Trigger.trigger_id]),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Model {self.model_id}>"
