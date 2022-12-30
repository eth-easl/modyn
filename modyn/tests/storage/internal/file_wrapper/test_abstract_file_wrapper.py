import pytest

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper


def test_init():
    file_wrapper = AbstractFileWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    assert file_wrapper.file_path == 'test'


def test_get_size():
    file_wrapper = AbstractFileWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_wrapper.get_size()


def test_get_samples():
    file_wrapper = AbstractFileWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_wrapper.get_samples(0, 1)


def test_get_sample():
    file_wrapper = AbstractFileWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_wrapper.get_sample(0)
