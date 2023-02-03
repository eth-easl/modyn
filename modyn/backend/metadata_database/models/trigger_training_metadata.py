"""TriggerTrainingMetadata model."""

from modyn.backend.metadata_database.metadata_base import MetadataBase
from sqlalchemy import Column, Double, ForeignKey, Integer


class TriggerTrainingMetadata(MetadataBase):
    """TriggerTrainingMetadata model.

    Metadata per trigger. This data is added by both the TrainerServer and the Selector.

    Additional required columns need to be added here.
    """

    __tablename__ = "trigger_training_metadata"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}
    trigger_training_metadata_id = Column("trigger_training_metadata_id", Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.pipeline_id"), nullable=False)
    trigger_id = Column("trigger_id", Integer, nullable=False)
    time_to_train = Column("time_to_train", Double)
    overall_loss = Column("overall_loss", Double)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<TriggerTrainingMetadata {self.pipeline_id}:{self.trigger_id}>"
