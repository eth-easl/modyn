import pytest

from modyn.storage.internal.database.storage_database_utils import get_filesystem_wrapper, get_file_wrapper
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import InvalidFilesystemWrapperTypeException
from modyn.storage.internal.file_wrapper.file_wrapper_type import InvalidFileWrapperTypeException


def test_get_filesystem_wrapper():
    filesystem_wrapper = get_filesystem_wrapper('local', '/tmp/modyn')
    assert filesystem_wrapper is not None
    assert filesystem_wrapper.base_path == '/tmp/modyn'
    assert filesystem_wrapper.type == 'local'


def test_get_filesystem_wrapper_with_invalid_type():
    with pytest.raises(InvalidFilesystemWrapperTypeException):
        filesystem_wrapper = get_filesystem_wrapper('invalid', '/tmp/modyn')
        assert filesystem_wrapper is None


def test_get_file_wrapper():
    file_wrapper = get_file_wrapper('mnist_webdataset', '/tmp/modyn')
    assert file_wrapper is not None
    assert file_wrapper.path == '/tmp/modyn'
    assert file_wrapper.type == 'local'


def test_get_file_wrapper_with_invalid_type():
    with pytest.raises(InvalidFileWrapperTypeException):
        file_wrapper = get_file_wrapper('invalid', '/tmp/modyn')
        assert file_wrapper is None
