"""Pipeline model."""

from modyn.metadata_database.metadata_base import MetadataBase
from sqlalchemy import Boolean, Column, Integer, String


class Pipeline(MetadataBase):
    """Pipeline model."""

    __tablename__ = "pipelines"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}
    pipeline_id = Column("pipeline_id", Integer, primary_key=True)
    num_workers = Column("num_workers", Integer, nullable=False)
    model_id = Column("model_id", String(length=50), nullable=False)
    model_config = Column("model_config", String(length=500), nullable=False)
    amp = Column("amp", Boolean, nullable=False)
    full_model_strategy_name = Column("full_model_strategy_name", String(length=50), nullable=False)
    full_model_strategy_zip = Column("full_model_strategy_zip", Boolean, default=None, nullable=True)
    full_model_strategy_zip_algorithm = Column(
        "full_model_strategy_zip_algorithm", String(length=50), default=None, nullable=True
    )
    full_model_strategy_config = Column("full_model_strategy_config", String(length=500), default=None, nullable=True)
    inc_model_strategy_name = Column("inc_model_strategy_name", String(length=50), default=None, nullable=True)
    inc_model_strategy_zip = Column("inc_model_strategy_zip", Boolean, default=None, nullable=True)
    inc_model_strategy_zip_algorithm = Column(
        "inc_model_strategy_zip_algorithm", String(length=50), default=None, nullable=True
    )
    inc_model_strategy_config = Column("inc_model_strategy_config", String(length=500), default=None, nullable=True)
    full_model_interval = Column("full_model_interval", Integer, default=None, nullable=True)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Pipeline {self.pipeline_id}>"
