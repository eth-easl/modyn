"""Sample model."""

from modyn.storage.internal.database.storage_base import Base
from sqlalchemy import BigInteger, Column, ForeignKey, Integer, String
from sqlalchemy.orm import backref, relationship


class Sample(Base):
    """Sample model."""

    __tablename__ = "samples"
    sample_id = Column("sample_id", Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.file_id"), nullable=False)
    file = relationship("File", backref=backref("samples", lazy=True))
    external_key = Column(String(120), unique=True, nullable=False)
    index = Column(BigInteger, nullable=False)
    label = Column(BigInteger, nullable=True)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Sample {self.sample_id}>"
