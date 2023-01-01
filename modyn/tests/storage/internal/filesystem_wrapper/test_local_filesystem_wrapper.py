import os
import pytest

from modyn.storage.internal.filesystem_wrapper.local_filesystem_wrapper import LocalFilesystemWrapper

test_dir = os.path.sep + os.path.join('tmp', 'modyn', 'test_dir')
test_file = os.path.sep + os.path.join('tmp', 'modyn', 'test_dir', 'test_file')
TEST_FILE_MODIFIED_AT = None
test_dir2 = os.path.sep + os.path.join('tmp', 'modyn', 'test_dir', 'test_dir')
test_file2 = os.path.sep + os.path.join('tmp', 'modyn', 'test_dir', 'test_dir', 'test_file')
TEST_FILE2_MODIFIED_AT = None


def setup():
    os.makedirs(test_dir, exist_ok=True)

    with open(test_file, 'w', encoding='utf8') as file:
        file.write('test1')

    global TEST_FILE_MODIFIED_AT  #  pylint: disable=global-statement # noqa: E262
    TEST_FILE_MODIFIED_AT = os.path.getmtime(test_file)

    os.makedirs(test_dir2, exist_ok=True)

    with open(test_file2, 'w', encoding='utf8') as file:
        file.write('test2 long')

    global TEST_FILE2_MODIFIED_AT  #  pylint: disable=global-statement # noqa: E262
    TEST_FILE2_MODIFIED_AT = os.path.getmtime(test_file2)


def teardown():
    os.remove(test_file)
    os.remove(test_file2)
    os.rmdir(test_dir2)
    os.rmdir(test_dir)


def test_init():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.base_dir == test_dir


def test_get_files():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    file = filesystem_wrapper.get_file(test_file)
    assert file == 'test1'

    file = filesystem_wrapper.get_file(test_file2)
    assert file == 'test2 long'


def test_get_file_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(FileNotFoundError):
        filesystem_wrapper.get_file('not_found')


def test_get_file_directory():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.get_file(test_dir2)


def test_get_file_not_in_base_dir():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_file(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_dir'))


def test_exists():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.exists(test_file)
    assert filesystem_wrapper.exists(test_file2)
    assert filesystem_wrapper.exists(test_dir)
    assert filesystem_wrapper.exists(test_dir2)


def test_exists_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.exists('not_found')


def test_exists_not_in_base_dir():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.exists(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_dir'))


def test_list():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.list(test_dir) == ['test_file', 'test_dir']
    assert filesystem_wrapper.list(test_dir2) == ['test_file']


def test_list_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(FileNotFoundError):
        filesystem_wrapper.list('not_found')


def test_list_not_directory():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(NotADirectoryError):
        filesystem_wrapper.list(test_file)


def test_list_not_in_base_dir():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.list(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_dir'))


def test_list_recursive():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.list(test_dir, recursive=True) == ['test_file', 'test_dir', 'test_dir' +
                                                                 os.path.sep + 'test_file']


def test_list_recursive_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(FileNotFoundError):
        filesystem_wrapper.list('not_found', recursive=True)


def test_list_recursive_not_directory():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(NotADirectoryError):
        filesystem_wrapper.list(test_file, recursive=True)


def test_list_recursive_not_in_base_dir():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.list(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_dir'), recursive=True)


def test_isdir():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.isdir(test_dir)
    assert filesystem_wrapper.isdir(test_dir2)


def test_isdir_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.isdir('not_found')


def test_isdir_not_directory():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.isdir(test_file)


def test_isdir_not_in_base_dir():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.isdir(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_dir'))


def test_isfile():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.isfile(test_file)
    assert filesystem_wrapper.isfile(test_file2)


def test_isfile_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.isfile('not_found')


def test_isfile_not_directory():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.isfile(test_dir)


def test_isfile_not_in_base_dir():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.isfile(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_dir'))


def test_getsize():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.getsize(test_file) == 5
    assert filesystem_wrapper.getsize(test_file2) == 9


def test_getsize_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(FileNotFoundError):
        filesystem_wrapper.getsize('not_found')


def test_getsize_not_file():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.getsize(test_dir)


def test_getsize_not_in_base_dir():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.getsize(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_dir'))


def test_getmodified():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.getmodified(test_file) == TEST_FILE_MODIFIED_AT
    assert filesystem_wrapper.getmodified(test_file2) == TEST_FILE2_MODIFIED_AT


def test_getmodified_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(FileNotFoundError):
        filesystem_wrapper.getmodified('not_found')


def test_getmodified_not_file():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.getmodified(test_dir)


def test_getmodified_not_in_base_dir():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.getmodified(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_dir'))


def test_getcreated():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.getcreated(test_file) == TEST_FILE_MODIFIED_AT
    assert filesystem_wrapper.getcreated(test_file2) == TEST_FILE2_MODIFIED_AT


def test_getcreated_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(FileNotFoundError):
        filesystem_wrapper.getcreated('not_found')


def test_getcreated_not_file():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.getcreated(test_dir)


def test_getcreated_not_in_base_dir():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.getcreated(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_dir'))


def test_join():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.join('a', 'b') == 'a' + os.path.sep + 'b'
    assert filesystem_wrapper.join('a', 'b', 'c') == 'a' + os.path.sep + 'b' + os.path.sep + 'c'
    assert filesystem_wrapper.join('a', 'b', 'c', 'd') == 'a' + \
        os.path.sep + 'b' + os.path.sep + 'c' + os.path.sep + 'd'
