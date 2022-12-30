import os
import typing
import datetime

from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FileSystemWrapperType


class LocalFilesystemWrapper(AbstractFileSystemWrapper):
    """
    Wrapper for local filesystem.
    """

    def __init__(self, base_path: str):
        super().__init__(base_path)
        self.filesystem_wrapper_type = FileSystemWrapperType.LOCAL

    def __is_valid_path(self, path: str) -> bool:
        return path.startswith(self.base_path)

    def get(self, path: str) -> bytes:
        if not self.__is_valid_path(path):
            raise ValueError(f'Path {path} is not valid.')
        if not self.isfile(path):
            raise IsADirectoryError(f'Path {path} is a directory.')
        with open(path, 'rb') as file:
            return file.read()

    def exists(self, path: str) -> bool:
        if not self.__is_valid_path(path):
            raise ValueError(f'Path {path} is not valid.')
        return os.path.exists(path)

    def list(self, path: str, recursive: bool = False) -> typing.List[str]:
        if not self.__is_valid_path(path):
            raise ValueError(f'Path {path} is not valid.')
        if recursive:
            return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn]
        return os.listdir(path)

    def isdir(self, path: str) -> bool:
        if not self.__is_valid_path(path):
            raise ValueError(f'Path {path} is not valid.')
        return os.path.isdir(path)

    def isfile(self, path: str) -> bool:
        if not self.__is_valid_path(path):
            raise ValueError(f'Path {path} is not valid.')
        return os.path.isfile(path)

    def get_size(self, path: str) -> int:
        if not self.__is_valid_path(path):
            raise ValueError(f'Path {path} is not valid.')
        return os.path.getsize(path)

    def get_modified(self, path: str) -> datetime.datetime:
        if not self.__is_valid_path(path):
            raise ValueError(f'Path {path} is not valid.')
        return datetime.datetime.fromtimestamp(os.path.getmtime(path))

    def get_created(self, path: str) -> datetime.datetime:
        if not self.__is_valid_path(path):
            raise ValueError(f'Path {path} is not valid.')
        return datetime.datetime.fromtimestamp(os.path.getctime(path))

    def join(self, *paths: str) -> str:
        return os.path.join(*paths)
