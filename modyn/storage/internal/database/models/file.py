from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship, backref

import datetime

from modyn.storage.internal.database import Base


class File(Base):
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('dataset.id'), nullable=False)
    dataset = relationship('Dataset', backref=backref('files', lazy=True))
    path = Column(String(120), unique=False, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    samples = relationship('Sample', backref='file', lazy=True)

    def __repr__(self):
        return '<DatasetFile %r>' % self.path

    def __init__(self, dataset, path):
        self.dataset = dataset
        self.path = path
        self.created_at = datetime.datetime.utcnow()
        self.updated_at = datetime.datetime.utcnow()
