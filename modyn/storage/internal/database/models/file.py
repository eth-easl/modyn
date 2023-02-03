"""File model."""

from modyn.storage.internal.database.storage_base import StorageBase
from sqlalchemy import BigInteger, Column, ForeignKey, Integer, String
from sqlalchemy.orm import backref, relationship


class File(StorageBase):
    """File model."""

    __tablename__ = "files"
    file_id = Column("file_id", Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"), nullable=False)
    dataset = relationship("Dataset", backref=backref("files", lazy=True))
    path = Column(String(120), unique=False, nullable=False)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)
    number_of_samples = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<File {self.path}>"
