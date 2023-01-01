from enum import Enum


class FileWrapperType(Enum):
    """Enum for the type of file system wrapper."""
    MNISTWebdatasetFileWrapper = 1  #  pylint: disable=invalid-name
    Parquet = 2  #  pylint: disable=invalid-name


class InvalidFileWrapperTypeException(Exception):
    """Invalid file wrapper type exception."""
    def __init__(self, message: str):
        super().__init__(message)
