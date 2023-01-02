from sqlalchemy import Column, String, Enum, Integer

from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType

from modyn.storage.internal.database.base import Base


class Dataset(Base):
    __tablename__ = 'dataset'
    id = Column(Integer, primary_key=True)
    name = Column(String(80), unique=True, nullable=False)
    description = Column(String(120), unique=False, nullable=True)
    version = Column(String(80), unique=False, nullable=True)
    filesystem_wrapper_type = Column(Enum(FilesystemWrapperType), nullable=False)
    file_wrapper_type = Column(Enum(FileWrapperType), nullable=False)
    base_path = Column(String(120), unique=False, nullable=False)

    def __repr__(self) -> str:
        return f'<Dataset {self.name}>'

    def __init__(self, name: str,
                 description: str,
                 filesystem_wrapper_type: FilesystemWrapperType,
                 file_wrapper_type: FileWrapperType,
                 base_path: str,
                 version: str = '0.0.1'):
        self.name = name
        self.description = description
        self.filesystem_wrapper_type = filesystem_wrapper_type
        self.file_wrapper_type = file_wrapper_type
        self.base_path = base_path
        self.version = version
