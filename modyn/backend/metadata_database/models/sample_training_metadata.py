"""SampleTrainingMetadata model."""

from modyn.backend.metadata_database.metadata_base import MetadataBase
from modyn.backend.metadata_database.models.triggers import Trigger
from sqlalchemy import Column, Double, Integer, String
from sqlalchemy.schema import ForeignKeyConstraint


class SampleTrainingMetadata(MetadataBase):
    """SampleTrainingMetadata model.

    # TODO(#65): Store the data accordingly in the database and extend the models if necessary.
    Metadata from training on a sample, collected by the MetadataCollector, sent to MetadataProcessor
    which stores it in the database.

    Additional required columns need to be added here.
    """

    __tablename__ = "sample_training_metadata"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    sample_training_metadata_id = Column("sample_training_metadata_id", Integer, primary_key=True)
    pipeline_id = Column("pipeline_id", Integer, nullable=False)
    trigger_id = Column("trigger_id", Integer, nullable=False)
    sample_key = Column("sample_key", String(120), nullable=False)
    loss = Column("loss", Double)
    gradient = Column("gradient", Double)
    __table_args__ = (
        ForeignKeyConstraint([pipeline_id, trigger_id], [Trigger.pipeline_id, Trigger.trigger_id]),
        {"extend_existing": True},
    )
