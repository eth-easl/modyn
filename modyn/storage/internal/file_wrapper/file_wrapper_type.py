from enum import Enum

class FileWrapperType(Enum):
    """Enum for the type of file system wrapper."""
    MNIST = 1
    PARQUET = 2
    