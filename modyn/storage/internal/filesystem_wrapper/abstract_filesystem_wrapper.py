"""Abstract filesystem wrapper class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType


class AbstractFileSystemWrapper(ABC):
    """Base class for all filesystem wrappers."""

    filesystem_wrapper_type: FilesystemWrapperType = None

    def __init__(self, base_path: str):
        """Init filesystem wrapper.

        Args:
            base_path (str): Base path of filesystem
        """
        self.base_path = base_path

    def get(self, path: Union[str, Path]) -> bytes:
        """Get file content.

        Args:
            path (Union[str, Path]): Absolute path to file

        Returns:
            bytes: File content
        """
        return self._get(str(path))

    @abstractmethod
    def _get(self, path: str) -> bytes:
        """Get file content.

        Args:
            path (str): Absolute path to file

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            bytes: File content
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Exists checks whether the given path exists or not.

        Args:
            path (str): Absolute path to file or directory

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            bool: True if path exists, False otherwise
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def list(self, path: str, recursive: bool = False) -> list[str]:
        """List files in directory.

        Args:
            path (str): Absolute path to directory
            recursive (bool, optional): Recursively list files. Defaults to False.

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            list[str]: List of files
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def isdir(self, path: str) -> bool:
        """Return `True` if the path is a directory.

        Args:
            path (str): Absolute path to file

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            bool: True if path is a directory, False otherwise
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def isfile(self, path: str) -> bool:
        """Return `True` if the path is a file.

        Args:
            path (str): Absolute path to file

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            bool: True if path is a file, False otherwise
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_size(self, path: str) -> int:
        """Return the size of the file.

        Args:
            path (str): Absolute path to file

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            int: Size of file
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_modified(self, path: str) -> int:
        """Return the last modified time of the file.

        Args:
            path (str): Absolute path to file

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            int: Last modified time
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_created(self, path: str) -> int:
        """Return the creation time of the file.

        Args:
            path (str): Absolute path to file

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            int: Creation time
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def join(self, *paths: str) -> str:
        """Join paths.

        Raises:
            NotImplementedError: If not implemented

        Returns:
            str: Joined path
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete file.

        Args:
            path (str): Absolute path to file

        Raises:
            NotImplementedError: If the method is not implemented
        """
        raise NotImplementedError
