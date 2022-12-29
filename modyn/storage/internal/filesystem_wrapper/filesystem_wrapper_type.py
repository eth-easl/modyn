from enum import Enum


class FileSystemWrapperType(Enum):
    """Enum for the type of file system wrapper."""
    LOCAL = 1
    S3 = 2


class InvalidFilesystemWrapperTypeException(Exception):
    """Exception for invalid filesystem wrapper type."""
    def __init__(self, message: str):
        super().__init__(message)
