from enum import Enum


class FileWrapperType(Enum):
    """Enum for the type of file system wrapper."""
    MNIST = 1
    PARQUET = 2


class InvalidFileWrapperTypeException(Exception):
    """Invalid file wrapper type exception."""
    def __init__(self, message: str):
        super().__init__(message)
