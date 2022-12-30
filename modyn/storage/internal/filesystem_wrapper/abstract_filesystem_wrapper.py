import datetime
import typing
from abc import ABC, abstractmethod

from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FileSystemWrapperType


class AbstractFileSystemWrapper(ABC):
    """
    Base class for all filesystem wrappers.
    """
    filesystem_wrapper_type: FileSystemWrapperType = None

    def __init__(self, base_path: str):
        self.base_path = base_path

    @abstractmethod
    def get(self, path: str) -> bytes:
        """
        Get the contents of a file.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Check if a file exists.
        """
        raise NotImplementedError

    @abstractmethod
    def list(self, path: str, recursive: bool = False) -> typing.List[str]:
        """
        List all files in a directory.
        """
        raise NotImplementedError

    @abstractmethod
    def isdir(self, path: str) -> bool:
        """
        Check if a path is a directory.
        """
        raise NotImplementedError

    @abstractmethod
    def isfile(self, path: str) -> bool:
        """
        Check if a path is a file.
        """
        raise NotImplementedError

    @abstractmethod
    def get_size(self, path: str) -> int:
        """
        Get the size of a file.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modified(self, path: str) -> datetime.datetime:
        """
        Get the last modified time of a file.
        """
        raise NotImplementedError

    @abstractmethod
    def get_created(self, path: str) -> datetime.datetime:
        """
        Get the creation time of a file.
        """
        raise NotImplementedError

    @abstractmethod
    def join(self, *paths: str) -> str:
        """
        Join paths.
        """
        raise NotImplementedError
