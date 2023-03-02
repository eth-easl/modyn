"""TriggerTrainingMetadata model."""

from modyn.backend.metadata_database.metadata_base import MetadataBase
from modyn.backend.metadata_database.models import Pipeline
from sqlalchemy import Column, ForeignKey, Integer


class Trigger(MetadataBase):
    """Trigger model."""

    __tablename__ = "triggers"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {
        "extend_existing": True,
    }
    trigger_id = Column("trigger_id", Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey(Pipeline.pipeline_id), nullable=False, primary_key=True)
    num_keys = Column("num_keys", Integer, nullable=True)
    num_partitions = Column("num_partitions", Integer, nullable=True)
