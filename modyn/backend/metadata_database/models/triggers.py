"""TriggerTrainingMetadata model."""

from modyn.backend.metadata_database.metadata_base import MetadataBase
from modyn.backend.metadata_database.models import Pipeline
from sqlalchemy import BigInteger, Column, ForeignKey, Index, Integer


class Trigger(MetadataBase):
    """Trigger model."""

    __tablename__ = "triggers"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    trigger_id = Column("trigger_id", Integer, primary_key=True)
    pipeline_id = Column("pipeline_id", Integer, ForeignKey(Pipeline.pipeline_id), nullable=False, primary_key=True)
    num_keys = Column("num_keys", BigInteger, nullable=True)
    num_partitions = Column("num_partitions", Integer, nullable=True)
    __table_args__ = (Index("t_trigger_pipeline_idx", "pipeline_id", "trigger_id"), {"extend_existing": True})
