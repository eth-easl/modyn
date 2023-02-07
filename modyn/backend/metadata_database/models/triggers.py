"""TriggerTrainingMetadata model."""

from modyn.backend.metadata_database.metadata_base import MetadataBase
from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship


class Trigger(MetadataBase):
    """Trigger model."""

    __tablename__ = "triggers"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}
    trigger_id = Column("trigger_id", Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.pipeline_id"), nullable=False)
    pipeline = relationship("Pipeline")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Trigger {self.pipeline_id}:{self.trigger_id}>"
