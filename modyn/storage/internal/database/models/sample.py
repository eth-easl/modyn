"""Sample model."""

from modyn.storage.internal.database.storage_base import StorageBase
from sqlalchemy import BigInteger, Column, ForeignKey, Integer, String
from sqlalchemy.orm import backref, relationship


class Sample(StorageBase):
    """Sample model."""

    __tablename__ = "samples"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing
    __table_args__ = {'extend_existing': True}
    sample_id = Column("sample_id", Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.file_id"), nullable=False)
    file = relationship("File", backref=backref("samples", lazy=True))
    external_key = Column(String(120), unique=True, nullable=False)
    index = Column(BigInteger, nullable=False)
    label = Column(BigInteger, nullable=True)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Sample {self.sample_id}>"
