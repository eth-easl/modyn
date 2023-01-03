import pytest

from modyn.storage.internal.database.storage_database_utils import get_filesystem_wrapper, get_file_wrapper
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import InvalidFilesystemWrapperTypeException, \
    FilesystemWrapperType
from modyn.storage.internal.file_wrapper.file_wrapper_type import InvalidFileWrapperTypeException, FileWrapperType


def test_get_filesystem_wrapper():
    filesystem_wrapper = get_filesystem_wrapper(FilesystemWrapperType.LocalFilesystemWrapper, '/tmp/modyn')
    assert filesystem_wrapper is not None
    assert filesystem_wrapper.base_path == '/tmp/modyn'
    assert filesystem_wrapper.filesystem_wrapper_type == FilesystemWrapperType.LocalFilesystemWrapper


def test_get_filesystem_wrapper_with_invalid_type():
    with pytest.raises(InvalidFilesystemWrapperTypeException):
        filesystem_wrapper = get_filesystem_wrapper('invalid', '/tmp/modyn')
        assert filesystem_wrapper is None


def test_get_file_wrapper():
    file_wrapper = get_file_wrapper(FileWrapperType.WebdatasetFileWrapper, '/tmp/modyn')
    assert file_wrapper is not None
    assert file_wrapper.file_wrapper_type == FileWrapperType.WebdatasetFileWrapper


def test_get_file_wrapper_with_invalid_type():
    with pytest.raises(InvalidFileWrapperTypeException):
        file_wrapper = get_file_wrapper('invalid', '/tmp/modyn')
        assert file_wrapper is None
