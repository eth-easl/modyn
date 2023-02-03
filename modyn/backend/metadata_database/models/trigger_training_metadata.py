"""TriggerTrainingMetadata model."""

from modyn.backend.metadata_database.metadata_base import MetadataBase
from sqlalchemy import Column, Double, Integer


class TriggerTrainingMetadata(MetadataBase):
    """TriggerTrainingMetadata model.

    Metadata per trigger. This data is added by both the TrainerServer and the Selector.

    Additional required columns need to be added here.
    """

    __tablename__ = "trigger_training_metadata"
    trigger_training_metadata_id = Column("trigger_training_metadata_id", Integer, primary_key=True)
    pipeline_id = Column("pipeline_id", Integer, nullable=False)
    trigger_id = Column("trigger_id", Integer, nullable=False)
    time_to_train = Column("time_to_train", Double)
    overall_loss = Column("overall_loss", Double)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<TriggerTrainingMetadata {self.pipeline_id}:{self.trigger_id}>"
