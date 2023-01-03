import datetime
import typing
from abc import ABC, abstractmethod

from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType


class AbstractFileSystemWrapper(ABC):
    """
    Base class for all filesystem wrappers.
    """
    filesystem_wrapper_type: FilesystemWrapperType = None

    def __init__(self, base_path: str):
        """
        The `base_path` is the base path of the dataset.

        It is expected that the constructor also initializes the filesystem_wrapper_type.
        """
        self.base_path = base_path

    @abstractmethod
    def get(self, path: str) -> bytes:
        """
        Returns the file object at the given path.

        The path is absolute and starts with the base path of the dataset.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Returns `True` if the file exists at the given path.

        The path is absolute and starts with the base path of the dataset.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def list(self, path: str, recursive: bool = False) -> typing.List[str]:
        """
        Returns a list of files in the given path.

        The path is absolute and starts with the base path of the dataset.
        If `recursive` is `True`, the list should be recursively iterated through
        all subdirectories. The list should contain the absolute paths of the files.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def isdir(self, path: str) -> bool:
        """
        Returns `True` if the path is a directory.

        The path is absolute and starts with the base path of the dataset.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def isfile(self, path: str) -> bool:
        """
        Returns `True` if the path is a file.

        The path is absolute and starts with the base path of the dataset.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_size(self, path: str) -> int:
        """
        Returns the size of the file.

        The path is absolute and starts with the base path of the dataset.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_modified(self, path: str) -> datetime.datetime:
        """
        Returns the last modified time of the file.

        The path is absolute and starts with the base path of the dataset.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_created(self, path: str) -> datetime.datetime:
        """
        Returns the creation time of the file.

        The path is absolute and starts with the base path of the dataset.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def join(self, *paths: str) -> str:
        """
        Joins the given paths together.
        """
        raise NotImplementedError  # pragma: no cover
