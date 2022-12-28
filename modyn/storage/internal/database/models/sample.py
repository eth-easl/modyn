from sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship, backref

from modyn.storage.internal.database import Base


class Sample(Base):
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('file.id'), nullable=False)
    file = relationship('File', backref=backref('samples', lazy=True))
    external_key = Column(String(120), unique=True, nullable=False)
    index = Column(Integer, nullable=False)

    def __repr__(self):
        return '<DatasetFileSample %r>' % self.id

    def __init__(self, file, external_key, index):
        self.file = file
        self.external_key = external_key
        self.index = index
