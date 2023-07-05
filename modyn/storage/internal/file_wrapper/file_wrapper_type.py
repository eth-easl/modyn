"""File wrapper type enum and exception."""

from enum import Enum


class FileWrapperType(Enum):
    """Enum for the type of file wrapper.

    Important: The value of the enum must be the same as the name of the module.
    The name of the enum must be the same as the name of the class.
    """

    SingleSampleFileWrapper = "single_sample_file_wrapper"  # pylint: disable=invalid-name
    BinaryFileWrapper = "binary_file_wrapper"  # pylint: disable=invalid-name
    CsvFileWrapper = "csv_file_wrapper"  # pylint: disable=invalid-name


class InvalidFileWrapperTypeException(Exception):
    """Invalid file wrapper type exception."""

    def __init__(self, message: str):
        """Init exception.

        Args:
            message (str): Exception message
        """
        super().__init__(message)
