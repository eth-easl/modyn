import logging

from modyn.utils import dynamic_module_import
from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType, InvalidFileWrapperTypeException
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FileSystemWrapperType, \
    InvalidFilesystemWrapperTypeException
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper

logger = logging.getLogger(__name__)


def get_filesystem_wrapper(filesystem_wrapper_type: FileSystemWrapperType, base_path: str) -> AbstractFileSystemWrapper:
    """
    Get the filesystem wrapper.
    """
    if not isinstance(filesystem_wrapper_type, FileSystemWrapperType):
        raise InvalidFilesystemWrapperTypeException('Invalid filesystem wrapper type.')
    filesystem_wrapper_module = \
        dynamic_module_import(f'modyn.storage.internal.filesystem_wrapper.{filesystem_wrapper_type.name.lower()}_filesystem_wrapper')
    filesystem_wrapper = getattr(
        filesystem_wrapper_module,
        f'{filesystem_wrapper_type.name.capitalize()}FilesystemWrapper'
    )
    return filesystem_wrapper(base_path)


def get_file_wrapper(file_wrapper_type: FileWrapperType, path: str) -> AbstractFileWrapper:
    """
    Get the file wrapper.
    """
    if not isinstance(file_wrapper_type, FileWrapperType):
        raise InvalidFileWrapperTypeException('Invalid file wrapper type.')
    file_wrapper_module = \
        dynamic_module_import(f'modyn.storage.internal.file_wrapper.{file_wrapper_type.name.lower()}_file_wrapper')
    file_wrapper = getattr(
        file_wrapper_module,
        f'{file_wrapper_type.name.capitalize()}FileWrapper'
    )
    return file_wrapper(path)
