import pytest
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import (
    FilesystemWrapperType,
)


def test_invalid_filesystem_wrapper_type():
    with pytest.raises(ValueError):
        filesystem_wrapper_type = FilesystemWrapperType("invalid")
        assert filesystem_wrapper_type is None
