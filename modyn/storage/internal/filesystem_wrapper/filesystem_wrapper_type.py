from enum import Enum


class FilesystemWrapperType(Enum):
    """
    Enum for the type of file system wrapper.
    Important: The value of the enum must be the same as the name of the module.
    The name of the enum must be the same as the name of the class.
    """
    LocalFilesystemWrapper = 'local_filesystem_wrapper'  # pylint: disable=invalid-name
    S3FileSystemWrapper = 's3_filesystem_wrapper'  # pylint: disable=invalid-name


class InvalidFilesystemWrapperTypeException(Exception):
    """Exception for invalid filesystem wrapper type."""

    def __init__(self, message: str):
        super().__init__(message)
