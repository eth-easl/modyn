from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship, backref

import datetime

from modyn.storage.internal.database.base import Base
from modyn.storage.internal.database.models.dataset import Dataset


class File(Base):
    __tablename__ = 'file'
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('dataset.id'), nullable=False)
    dataset = relationship('Dataset', backref=backref('files', lazy=True))
    path = Column(String(120), unique=False, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    number_of_samples = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f'<File {self.path}>'

    def __init__(self,
                 dataset: Dataset,
                 path: str,
                 created_at:
                 datetime.datetime,
                 updated_at: datetime.datetime,
                 number_of_samples: int):
        self.dataset = dataset
        self.path = path
        self.created_at = created_at
        self.updated_at = updated_at
        self.number_of_samples = number_of_samples
