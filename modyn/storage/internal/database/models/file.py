from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship, backref

import datetime

from modyn.storage.internal.database.base import Base
from modyn.storage.internal.database.models.dataset import Dataset


class File(Base):
    dataset_id = Column(Integer, ForeignKey('dataset.id'), nullable=False)
    dataset = relationship('Dataset', backref=backref('files', lazy=True))
    path = Column(String(120), unique=False, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    samples = relationship('Sample', backref='file', lazy=True)

    def __repr__(self) -> str:
        return f'<DatasetFile {self.path}>'

    def __init__(self,
                 dataset: Dataset,
                 path: str,
                 created_at:
                 datetime.datetime,
                 updated_at: datetime.datetime):
        self.dataset = dataset
        self.path = path
        self.created_at = created_at
        self.updated_at = updated_at
