from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType


class ParquetFileWrapper(AbstractFileWrapper):
    """Parquet file wrapper."""

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_wrapper_type = FileWrapperType.PARQUET

    def get_size(self) -> int:
        """Get the size of the file."""
        raise NotImplementedError

    def get_samples(self, start: int, end: int) -> bytes:
        """Get samples from the file."""
        raise NotImplementedError

    def get_sample(self, index: int) -> bytes:
        """Get a sample from the file."""
        raise NotImplementedError
