# pylint: disable=super-init-not-called
"""File model."""

from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.storage_base import Base
from sqlalchemy import BigInteger, Column, ForeignKey, Integer, String
from sqlalchemy.orm import backref, relationship


class File(Base):
    """File model."""

    __tablename__ = "files"
    file_id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"), nullable=False)
    dataset = relationship("Dataset", backref=backref("files", lazy=True))
    path = Column(String(120), unique=False, nullable=False)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)
    number_of_samples = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<File {self.path}>"

    def __init__(self, dataset: Dataset, path: str, created_at: int, updated_at: int, number_of_samples: int):
        """Init file.

        Args:
            dataset (Dataset): dataset reference
            path (str): path
            created_at (int): created at
            updated_at (int): updated at
            number_of_samples (int): number of samples
        """
        self.dataset = dataset
        self.path = path
        self.created_at = created_at
        self.updated_at = updated_at
        self.number_of_samples = number_of_samples
