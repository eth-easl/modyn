import pytest

from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper


def test_init():
    file_system_wrapper = AbstractFileSystemWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    assert file_system_wrapper.base_path == 'test'


def test_get():
    file_system_wrapper = AbstractFileSystemWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_system_wrapper.get('test')


def test_exists():
    file_system_wrapper = AbstractFileSystemWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_system_wrapper.exists('test')


def test_list():
    file_system_wrapper = AbstractFileSystemWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_system_wrapper.list('test')


def test_isdir():
    file_system_wrapper = AbstractFileSystemWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_system_wrapper.isdir('test')


def test_isfile():
    file_system_wrapper = AbstractFileSystemWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_system_wrapper.isfile('test')


def test_get_size():
    file_system_wrapper = AbstractFileSystemWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_system_wrapper.get_size('test')


def test_get_modified():
    file_system_wrapper = AbstractFileSystemWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_system_wrapper.get_modified('test')


def test_get_created():
    file_system_wrapper = AbstractFileSystemWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_system_wrapper.get_created('test')


def test_join():
    file_system_wrapper = AbstractFileSystemWrapper('test')  # pylint: disable=abstract-class-instantiated # noqa: E262
    with pytest.raises(NotImplementedError):
        file_system_wrapper.join('test')
