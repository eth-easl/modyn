from enum import Enum


class FileWrapperType(Enum):
    """
    Enum for the type of file wrapper.
    Important: The value of the enum must be the same as the name of the module.
    The name of the enum must be the same as the name of the class.
    """
    WebdatasetFileWrapper = 'webdataset_file_wrapper'  # pylint: disable=invalid-name
    ParquetFileWrapper = 'parquet_file_wrapper'  # pylint: disable=invalid-name
    SimpleFileWrapper = 'simple_file_wrapper'  # pylint: disable=invalid-name


class InvalidFileWrapperTypeException(Exception):
    """Invalid file wrapper type exception."""

    def __init__(self, message: str):
        super().__init__(message)
