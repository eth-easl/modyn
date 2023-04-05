"""Local filesystem wrapper.

This module contains the local filesystem wrapper.
It is used to access files on the local filesystem.
"""
import os

from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType


class LocalFilesystemWrapper(AbstractFileSystemWrapper):
    """Local filesystem wrapper."""

    def __init__(self, base_path: str):
        """Init local filesystem wrapper.

        Args:
            base_path (str): Base path of local filesystem
        """
        super().__init__(base_path)
        self.filesystem_wrapper_type = FilesystemWrapperType.LocalFilesystemWrapper

    def __is_valid_path(self, path: str) -> bool:
        return path.startswith(self.base_path)

    def _get(self, path: str) -> bytes:
        """Get file content.

        Args:
            path (str): Absolute path to file

        Raises:
            FileNotFoundError: If path is not valid
            IsADirectoryError: If path is a directory

        Returns:
            bytes: File content
        """
        if not self.__is_valid_path(path):
            raise ValueError(f"Path {path} is not valid.")
        if not self.isfile(path):
            raise IsADirectoryError(f"Path {path} is a directory.")
        with open(path, "rb") as file:
            return file.read()

    def exists(self, path: str) -> bool:
        """Check if path exists.

        Args:
            path (str): Absolute path to file or directory

        Returns:
            bool: True if path exists, False otherwise
        """
        return os.path.exists(path)

    def list(self, path: str, recursive: bool = False) -> list[str]:
        """List files in directory.

        Args:
            path (str): Absolute path to directory
            recursive (bool, optional): List files recursively. Defaults to False.

        Raises:
            ValueError: If path is not valid
            NotADirectoryError: If path is not a directory

        Returns:
            list[str]: List of files in directory
        """
        if not self.__is_valid_path(path):
            raise ValueError(f"Path {path} is not valid.")
        if not self.isdir(path):
            raise NotADirectoryError(f"Path {path} is not a directory.")
        if recursive:
            return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn]
        return os.listdir(path)

    def isdir(self, path: str) -> bool:
        """Check if path is a directory.

        Args:
            path (str): Absolute path to directory

        Returns:
            bool: True if path is a directory, False otherwise
        """
        return os.path.isdir(path)

    def isfile(self, path: str) -> bool:
        """Check if path is a file.

        Args:
            path (str): Absolute path to file

        Returns:
            bool: True if path is a file, False otherwise
        """
        return os.path.isfile(path)

    def get_size(self, path: str) -> int:
        """Get size of file.

        Args:
            path (str): Absolute path to file

        Raises:
            ValueError: If path is not valid
            IsADirectoryError: If path is a directory

        Returns:
            int: Size of file in bytes
        """
        if not self.__is_valid_path(path):
            raise ValueError(f"Path {path} is not valid.")
        if not self.isfile(path):
            raise IsADirectoryError(f"Path {path} is a directory.")
        return os.path.getsize(path)

    def get_modified(self, path: str) -> int:
        """Get modification time of file.

        Args:
            path (str): Absolute path to file

        Raises:
            ValueError: If path is not valid
            IsADirectoryError: If path is a directory

        Returns:
            int: Modification time in milliseconds rounded to the nearest integer
        """
        if not self.__is_valid_path(path):
            raise ValueError(f"Path {path} is not valid.")
        if not self.isfile(path):
            raise IsADirectoryError(f"Path {path} is a directory.")
        return int(os.path.getmtime(path) * 1000)

    def get_created(self, path: str) -> int:
        """Get creation time of file.

        Args:
            path (str): Absolute path to file

        Raises:
            ValueError: If path is not valid
            IsADirectoryError: If path is a directory

        Returns:
            int: Creation time in milliseconds rounded to the nearest integer
        """
        if not self.__is_valid_path(path):
            raise ValueError(f"Path {path} is not valid.")
        if not self.isfile(path):
            raise IsADirectoryError(f"Path {path} is a directory.")
        return int(os.path.getctime(path) * 1000)

    def join(self, *paths: str) -> str:
        """Join paths.

        Returns:
            str: Joined path
        """
        return os.path.join(*paths)

    def delete(self, path: str) -> None:
        """Delete file.

        Args:
            path (str): Absolute path to file

        Raises:
            ValueError: If path is not valid
            IsADirectoryError: If path is a directory
        """
        return os.remove(path)
