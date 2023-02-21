"""Dataset model."""

from modyn.storage.internal.database.storage_base import StorageBase
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType
from sqlalchemy import Column, Enum, Integer, String


class Dataset(StorageBase):
    """Dataset model."""

    __tablename__ = "datasets"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}
    dataset_id = Column("dataset_id", Integer, primary_key=True)
    name = Column(String(80), unique=True, nullable=False)
    description = Column(String(120), unique=False, nullable=True)
    version = Column(String(80), unique=False, nullable=True)
    filesystem_wrapper_type = Column(Enum(FilesystemWrapperType), nullable=False)
    file_wrapper_type = Column(Enum(FileWrapperType), nullable=False)
    base_path = Column(String(120), unique=False, nullable=False)
    file_wrapper_config = Column(String(240), unique=False, nullable=True)
    last_timestamp = Column(Integer, unique=False, nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Dataset {self.name}>"
