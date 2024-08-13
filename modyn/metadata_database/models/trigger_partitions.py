from sqlalchemy import BigInteger, Column, ForeignKey, ForeignKeyConstraint, Integer

from modyn.metadata_database.metadata_base import MetadataBase
from modyn.metadata_database.models import Pipeline


class TriggerPartition(MetadataBase):
    """TriggerPartition model."""

    __tablename__ = "trigger_partitions"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    trigger_id = Column("trigger_id", Integer, nullable=False, primary_key=True)
    pipeline_id = Column("pipeline_id", Integer, ForeignKey(Pipeline.pipeline_id), nullable=False, primary_key=True)
    partition_id = Column("partition_id", Integer, nullable=False, primary_key=True)
    num_keys = Column("num_keys", BigInteger, nullable=True)
    __table_args__ = (
        ForeignKeyConstraint(
            ["trigger_id", "pipeline_id"],
            ["triggers.trigger_id", "triggers.pipeline_id"],
        ),
        {"extend_existing": True},
    )
