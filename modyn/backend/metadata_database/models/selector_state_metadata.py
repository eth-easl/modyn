"""SelectorStateMetadata model."""

from modyn.backend.metadata_database.metadata_base import MetadataBase
from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, Integer, String


class SelectorStateMetadata(MetadataBase):
    """SelectorStateMetadata model.

    Metadata persistently stored by the Selector.

    Additional required columns need to be added here.
    """

    __tablename__ = "selector_state_metadata"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}
    selector_state_metadata_id = Column("selector_state_metadata_id", Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.pipeline_id"), nullable=False)
    sample_key = Column("sample_key", String(120), nullable=False)
    seen_in_trigger_id = Column("seen_in_trigger_id", Integer)
    used = Column("used", Boolean, default=False)
    timestamp = Column("timestamp", BigInteger)
    label = Column("label", Integer)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<SelectorStateMetadata {self.pipeline_id}:{self.sample_key}>"
