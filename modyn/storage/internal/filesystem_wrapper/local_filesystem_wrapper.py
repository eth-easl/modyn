import os
import typing
import datetime

from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FileSystemWrapperType


class LocalFSWrapper(AbstractFileSystemWrapper):
    """
    Wrapper for local filesystem.
    """

    def __init__(self, base_path: str):
        super().__init__(base_path)
        self.filesystem_wrapper_type = FileSystemWrapperType.LOCAL

    def get(self, path: str) -> bytes:
        with open(path, 'rb') as file:
            return file.read()

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def list(self, path: str) -> typing.List[str]:
        return os.listdir(path)

    def isdir(self, path: str) -> bool:
        return os.path.isdir(path)

    def isfile(self, path: str) -> bool:
        return os.path.isfile(path)

    def get_size(self, path: str) -> int:
        return os.path.getsize(path)

    def get_modified(self, path: str) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path))

    def get_created(self, path: str) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(os.path.getctime(path))

    def get_accessed(self, path: str) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(os.path.getatime(path))

    def join(self, *paths: str) -> str:
        return os.path.join(*paths)
