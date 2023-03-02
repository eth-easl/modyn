"""TriggerSampleTrainingMetadata model."""
from modyn.backend.metadata_database.metadata_base import MetadataBase
from modyn.backend.metadata_database.models.triggers import Trigger
from sqlalchemy import BigInteger, Column, Float, Integer, String
from sqlalchemy.dialects import sqlite
from sqlalchemy.schema import ForeignKeyConstraint

BIGINT = BigInteger().with_variant(sqlite.INTEGER(), "sqlite")


class TriggerSample(MetadataBase):
    """TriggerSample model.

    Relationship between samples and triggers. Using this table, we know which samples were used in which trigger.

    Additional required columns need to be added here.
    """

    __tablename__ = "trigger_sample_training_metadata"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    trigger_sample_list_id = Column("trigger_sample_training_metadata_id", BIGINT, autoincrement=True, primary_key=True)
    pipeline_id = Column("pipeline_id", Integer, nullable=False)
    trigger_id = Column("trigger_id", Integer, nullable=False)
    partition_id = Column("partition_id", Integer, nullable=False)
    sample_key = Column("sample_key", String(120), nullable=False)
    sample_weight = Column("sample_weight", Float, nullable=False)
    __table_args__ = (
        ForeignKeyConstraint([pipeline_id, trigger_id], [Trigger.pipeline_id, Trigger.trigger_id]),
        {"extend_existing": True},
    )
