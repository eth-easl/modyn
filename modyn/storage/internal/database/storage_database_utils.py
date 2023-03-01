"""Storage database utilities."""

import json
import logging
from typing import Optional, Type

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType, InvalidFileWrapperTypeException
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import (
    FilesystemWrapperType,
    InvalidFilesystemWrapperTypeException,
)
from modyn.utils import dynamic_module_import

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
        raise InvalidFilesystemWrapperTypeException("Invalid filesystem wrapper type.")
    filesystem_wrapper_module = dynamic_module_import(
        f"modyn.storage.internal.filesystem_wrapper.{filesystem_wrapper_type.value}"
    )
    filesystem_wrapper = getattr(filesystem_wrapper_module, f"{filesystem_wrapper_type.name}")
    return filesystem_wrapper(base_path)


def get_file_wrapper(
    file_wrapper_type: FileWrapperType,
    path: str,
    file_wrapper_config: str,
    filesystem_wrapper: AbstractFileSystemWrapper,
    forced_file_wrapper: Optional[Type] = None,
) -> AbstractFileWrapper:
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
        raise InvalidFileWrapperTypeException("Invalid file wrapper type.")
    file_wrapper_config = json.loads(file_wrapper_config)
    if forced_file_wrapper is None:
        file_wrapper_module = dynamic_module_import(f"modyn.storage.internal.file_wrapper.{file_wrapper_type.value}")
        file_wrapper = getattr(file_wrapper_module, f"{file_wrapper_type.name}")
    else:
        file_wrapper = forced_file_wrapper

    return file_wrapper(path, file_wrapper_config, filesystem_wrapper)
