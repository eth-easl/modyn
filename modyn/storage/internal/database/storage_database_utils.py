"""Storage database utilities."""

import logging
import json

from modyn.utils import dynamic_module_import
from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType, InvalidFileWrapperTypeException
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType, \
    InvalidFilesystemWrapperTypeException
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper

logger = logging.getLogger(__name__)


def get_filesystem_wrapper(filesystem_wrapper_type: FilesystemWrapperType, base_path: str) -> AbstractFileSystemWrapper:
    """Get the filesystem wrapper.

    Args:
        filesystem_wrapper_type (FilesystemWrapperType): filesystem wrapper type
        base_path (str): base path of the filesystem wrapper

    Raises:
        InvalidFilesystemWrapperTypeException: Invalid filesystem wrapper type.

    Returns:
        AbstractFileSystemWrapper: filesystem wrapper
    """
    if not isinstance(filesystem_wrapper_type, FilesystemWrapperType):
        raise InvalidFilesystemWrapperTypeException('Invalid filesystem wrapper type.')
    filesystem_wrapper_module = \
        dynamic_module_import(
            f'modyn.storage.internal.filesystem_wrapper.{filesystem_wrapper_type.value}')
    filesystem_wrapper = getattr(
        filesystem_wrapper_module,
        f'{filesystem_wrapper_type.name}'
    )
    return filesystem_wrapper(base_path)


def get_file_wrapper(file_wrapper_type: FileWrapperType, path: str, file_wrapper_config: str) -> AbstractFileWrapper:
    """Get the file wrapper.

    Args:
        file_wrapper_type (FileWrapperType): file wrapper type
        path (str): path of the file wrapper
        file_wrapper_config (str): file wrapper configuration as json string.


    Raises:
        InvalidFileWrapperTypeException: Invalid file wrapper type.

    Returns:
        AbstractFileWrapper: file wrapper
    """
    if not isinstance(file_wrapper_type, FileWrapperType):
        raise InvalidFileWrapperTypeException('Invalid file wrapper type.')
    file_wrapper_config = json.loads(file_wrapper_config)
    file_wrapper_module = \
        dynamic_module_import(f'modyn.storage.internal.file_wrapper.{file_wrapper_type.value}')
    file_wrapper = getattr(
        file_wrapper_module,
        f'{file_wrapper_type.name}'
    )
    return file_wrapper(path, file_wrapper_config)
