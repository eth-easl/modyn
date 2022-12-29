from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship, backref

from modyn.storage.internal.database.base import Base


class Sample(Base):
    file_id = Column(Integer, ForeignKey('file.id'), nullable=False)
    file = relationship('File', backref=backref('samples', lazy=True))
    external_key = Column(String(120), unique=True, nullable=False)
    index = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f'<DatasetFileSample {self.id}>'

    def __init__(self, file: str, external_key: str, index: int):
        self.file = file
        self.external_key = external_key
        self.index = index
