"""SelectorStateMetadata model."""

from modyn.backend.metadata_database.metadata_base import MetadataBase
from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, Index, Integer, String
from sqlalchemy.dialects import sqlite

BIGINT = BigInteger().with_variant(sqlite.INTEGER(), "sqlite")


class SelectorStateMetadata(MetadataBase):
    """SelectorStateMetadata model.

    Metadata persistently stored by the Selector.

    Additional required columns need to be added here.
    """

    __tablename__ = "selector_state_metadata"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    selector_state_metadata_id = Column("selector_state_metadata_id", BIGINT, autoincrement=True, primary_key=True)
    pipeline_id = Column("pipeline_id", Integer, ForeignKey("pipelines.pipeline_id"), nullable=False, index=True)
    sample_key = Column("sample_key", String(120), nullable=False)
    seen_in_trigger_id = Column("seen_in_trigger_id", Integer)
    used = Column("used", Boolean, default=False)
    timestamp = Column("timestamp", BigInteger)
    label = Column("label", Integer)
    # ssm_pipeline_seen_idx: Optimizes new data strategy
    __table_args__ = (Index("ssm_pipeline_seen_idx", "pipeline_id", "seen_in_trigger_id"), {"extend_existing": True})
