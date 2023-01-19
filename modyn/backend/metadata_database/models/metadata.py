# pylint: disable=too-many-instance-attributes
"""Metadata model."""

from modyn.backend.metadata_database.metadata_base import Base
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import backref, relationship


class Metadata(Base):
    """Metadata model."""

    __tablename__ = "metadata"
    id = Column(Integer, primary_key=True)
    key = Column(String(120), nullable=False, primary_key=True)
    score = Column(Float, nullable=False)
    seen = Column(Boolean, nullable=False)
    label = Column(Integer, nullable=False)
    data = Column(LargeBinary, nullable=False)
    training_id = Column(Integer, ForeignKey("trainings.id"), nullable=False, primary_key=True)
    training = relationship("Training", backref=backref("metadata", lazy=True))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Metadata {self.key}>"

    def __init__(self, key: str, score: float, seen: bool, label: int, data: bytes, training_id: int):
        """Init metadata.
        Args:
            key (str): key
            score (float): score
            seen (bool): seen
            label (int): label
            data (bytes): data
            training (Training): training reference
        """
        self.key = key
        self.score = score
        self.seen = seen
        self.label = label
        self.data = data
        self.training_id = training_id
