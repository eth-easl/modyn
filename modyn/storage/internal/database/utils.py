from modyn.utils import dynamic_module_import

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.file_system_wrapper.file_system_wrapper_type import FileSystemWrapperType
from modyn.storage.internal.file_system_wrapper.abstract_file_system_wrapper import AbstractFileSystemWrapper

def get_file_system_wrapper(self, file_system_wrapper_type: FileSystemWrapperType, base_path: str) -> AbstractFileSystemWrapper:
    """
    Get the filesystem wrapper.
    """
    file_system_wrapper_module = dynamic_module_import('storage.internal.file_system_wrapper')
    fs_wrapper = getattr(
        file_system_wrapper_module,
        file_system_wrapper_type.value)()
    if fs_wrapper is None:
        raise Exception('Invalid filesystem wrapper type.')
    return fs_wrapper(base_path)

def get_file_wrapper(self, file_wrapper_type: FileWrapperType, path: str) -> AbstractFileWrapper:
    """
    Get the file wrapper.
    """
    file_wrapper_module = dynamic_module_import('storage.internal.file_wrapper')
    file_wrapper = getattr(
        file_wrapper_module,
        file_wrapper_type.value)()
    if file_wrapper is None:
        raise Exception('Invalid file wrapper type.')
    return file_wrapper(path)