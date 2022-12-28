from sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, DateTime, Enum
from sqlalchemy.orm import relationship

import datetime

from modyn.storage.internal.file_system_wrapper.file_system_wrapper_type import FileSystemWrapperType
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileType

from modyn.storage.internal.database import Base


class Dataset(Base):
    id = Column(Integer, primary_key=True)
    name = Column(String(80), unique=True, nullable=False)
    description = Column(String(120), unique=False, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    filesystem_type = Column(Enum(FileSystemWrapperType), nullable=False)
    file_type = Column(Enum(FileType), nullable=False)
    base_path = Column(String(120), unique=False, nullable=False)
    files = relationship('File', backref='dataset', lazy=True)

    def __repr__(self):
        return '<Dataset %r>' % self.name

    def __init__(self, name, description, filesystem_type, file_type, base_path):
        self.name = name
        self.description = description
        self.filesystem_type = filesystem_type
        self.file_type = file_type
        self.base_path = base_path
        self.created_at = datetime.datetime.utcnow()
        self.updated_at = datetime.datetime.utcnow()
