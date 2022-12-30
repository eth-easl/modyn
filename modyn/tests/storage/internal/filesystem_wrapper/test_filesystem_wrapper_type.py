import pytest

from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import InvalidFilesystemWrapperTypeException,\
    FileSystemWrapperType


def test_invalid_filesystem_wrapper_type():
    with pytest.raises(InvalidFilesystemWrapperTypeException):
        filesystem_wrapper_type = FileSystemWrapperType('invalid')
        assert filesystem_wrapper_type is None
