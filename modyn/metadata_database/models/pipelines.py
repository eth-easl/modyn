"""Pipeline model."""

from modyn.metadata_database.metadata_base import MetadataBase
from sqlalchemy import Column, Integer, Text


class Pipeline(MetadataBase):
    """Pipeline model."""

    __tablename__ = "pipelines"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}
    pipeline_id = Column("pipeline_id", Integer, primary_key=True)
    num_workers = Column("num_workers", Integer, nullable=False)
    selection_strategy = Column("selection_strategy", Text, nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Pipeline {self.pipeline_id}>"
