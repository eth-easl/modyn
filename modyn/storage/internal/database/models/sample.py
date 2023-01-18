"""Sample model."""

from modyn.storage.internal.database.base import Base
from sqlalchemy import Column, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import backref, relationship


class Sample(Base):
    """Sample model."""

    __tablename__ = "samples"
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    file = relationship("File", backref=backref("samples", lazy=True))
    external_key = Column(String(120), unique=True, nullable=False)
    index = Column(Integer, nullable=False)
    label = Column(LargeBinary, nullable=True)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Sample {self.id}>"

    def __init__(self, file: str, external_key: str, index: int, label: bytes = b""):
        """Init sample.

        Args:
            file (str): file reference
            external_key (str): external key
            index (int): index
        """
        self.file = file
        self.external_key = external_key
        self.index = index
        if label != b"":
            self.label = label
