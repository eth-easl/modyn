"""SampleTrainingMetadata model."""

from modyn.backend.metadata_database.metadata_base import MetadataBase
from sqlalchemy import Column, Double, ForeignKey, Integer, String
from sqlalchemy.orm import relationship


class SampleTrainingMetadata(MetadataBase):
    """SampleTrainingMetadata model.

    # TODO(#65): Store the data accordingly in the database and extend the models if necessary.
    Metadata from training on a sample, collected by the MetadataCollector, sent to MetadataProcessor
    which stores it in the database.

    Additional required columns need to be added here.
    """

    __tablename__ = "sample_training_metadata"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}
    sample_training_metadata_id = Column("sample_training_metadata_id", Integer, primary_key=True)
    trigger_id = Column(Integer, ForeignKey("triggers.trigger_id"), nullable=False)
    trigger = relationship("Trigger")
    sample_key = Column("sample_key", String(120), nullable=False)
    loss = Column("loss", Double)
    gradient = Column("gradient", Double)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<SampleTrainingMetadata {self.trigger_id}:{self.sample_key}>"
