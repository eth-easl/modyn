"""TriggerTrainingMetadata model."""

from modyn.backend.metadata_database.metadata_base import MetadataBase
from modyn.backend.metadata_database.models.triggers import Trigger
from sqlalchemy import Column, Double, Integer
from sqlalchemy.schema import ForeignKeyConstraint


class TriggerTrainingMetadata(MetadataBase):
    """TriggerTrainingMetadata model.

    Metadata per trigger. This data is added by both the TrainerServer and the Selector.

    Additional required columns need to be added here.
    """

    __tablename__ = "trigger_training_metadata"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    trigger_training_metadata_id = Column("trigger_training_metadata_id", Integer, primary_key=True)
    trigger_id = Column("trigger_id", Integer, nullable=False)
    pipeline_id = Column("pipeline_id", Integer, nullable=False)
    time_to_train = Column("time_to_train", Double)
    overall_loss = Column("overall_loss", Double)
    __table_args__ = (
        ForeignKeyConstraint([pipeline_id, trigger_id], [Trigger.pipeline_id, Trigger.trigger_id]),
        {"extend_existing": True},
    )
