import pytest

from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType, InvalidFileWrapperTypeException


def test_invalid_file_wrapper_type():
    with pytest.raises(InvalidFileWrapperTypeException):
        file_wrapper_type = FileWrapperType('invalid')
        assert file_wrapper_type is None
