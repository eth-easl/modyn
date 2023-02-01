# pylint: disable=too-many-instance-attributes, super-init-not-called
"""Metadata model."""

from modyn.backend.metadata_database.metadata_base import Base
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import backref, relationship


class Metadata(Base):
    """Metadata model."""

    __tablename__ = "metadata"
    metadata_id = Column("metadata_id", Integer, primary_key=True)
    key = Column(String(120), nullable=False)
    timestamp = Column(Integer, nullable=False)
    score = Column(Float, nullable=True)
    seen = Column(Boolean, nullable=True)
    label = Column(Integer, nullable=True)
    data = Column(LargeBinary, nullable=True)
    pipeline_id = Column(
        Integer,
        # TODO(#113): We do not have "trainings" anymore, only pipelines/triggers
        ForeignKey("trainings.training_id"),
        nullable=False,
    )
    training = relationship("Training", backref=backref("metadata", lazy=True))
    trigger_id = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Metadata {self.key}>"
