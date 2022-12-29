from modyn.utils import dynamic_module_import

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType, InvalidFileWrapperTypeException
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FileSystemWrapperType, \
    InvalidFilesystemWrapperTypeException
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper


def get_filesystem_wrapper(filesystem_wrapper_type: FileSystemWrapperType, base_path: str) -> AbstractFileSystemWrapper:
    """
    Get the filesystem wrapper.
    """
    filesystem_wrapper_module = dynamic_module_import('storage.internal.filesystem_wrapper')
    fs_wrapper = getattr(
        filesystem_wrapper_module,
        filesystem_wrapper_type.value)()
    if fs_wrapper is None:
        raise InvalidFilesystemWrapperTypeException('Invalid filesystem wrapper type.')
    return fs_wrapper(base_path)


def get_file_wrapper(file_wrapper_type: FileWrapperType, path: str) -> AbstractFileWrapper:
    """
    Get the file wrapper.
    """
    file_wrapper_module = dynamic_module_import('storage.internal.file_wrapper')
    file_wrapper = getattr(
        file_wrapper_module,
        file_wrapper_type.value)()
    if file_wrapper is None:
        raise InvalidFileWrapperTypeException('Invalid file wrapper type.')
    return file_wrapper(path)
