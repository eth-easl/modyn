"""TriggerSampleTrainingMetadata model."""
from modyn.backend.metadata_database.metadata_base import MetadataBase
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship


class TriggerSampleTrainingMetadata(MetadataBase):
    """TriggerSampleTrainingMetadata model.

    Metadata per sample in a trigger. This data is added by both the TrainerServer and the Selector.

    Additional required columns need to be added here.
    """

    __tablename__ = "trigger_sample_training_metadata"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}
    trigger_sample_training_metadata_id = Column("trigger_sample_training_metadata_id", Integer, primary_key=True)
    trigger_training_metadata_id = Column(
        Integer, ForeignKey("trigger_training_metadata.trigger_training_metadata_id"), nullable=False
    )
    trigger_training_metadata = relationship("TriggerTrainingMetadata")
    sample_key = Column("sample_key", String(120), nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<TriggerSampleTrainingMetadata {self.trigger_training_metadata_id}:{self.sample_key}>"
