"""File model."""

from modyn.storage.internal.database.storage_base import StorageBase
from sqlalchemy import BigInteger, Column, ForeignKey, Integer, String
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import relationship

BIGINT = BigInteger().with_variant(sqlite.INTEGER(), "sqlite")


class File(StorageBase):
    """File model."""

    __tablename__ = "files"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}
    file_id = Column("file_id", BIGINT, autoincrement=True, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"), nullable=False)
    dataset = relationship("Dataset")
    path = Column(String(120), unique=False, nullable=False)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)
    number_of_samples = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<File {self.path}>"
