from enum import Enum

class FileSystemWrapperType(Enum):
    """Enum for the type of file system wrapper."""
    LOCAL = 1
    S3 = 2