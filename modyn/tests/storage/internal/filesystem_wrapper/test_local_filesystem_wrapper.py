import os
import pytest
import datetime

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
    TEST_FILE_MODIFIED_AT = datetime.datetime.fromtimestamp(os.path.getmtime(test_file))

    os.makedirs(test_dir2, exist_ok=True)

    with open(test_file2, 'w', encoding='utf8') as file:
        file.write('test2 long')

    global TEST_FILE2_MODIFIED_AT  #  pylint: disable=global-statement # noqa: E262
    TEST_FILE2_MODIFIED_AT = datetime.datetime.fromtimestamp(os.path.getmtime(test_file2))


def teardown():
    os.remove(test_file)
    os.remove(test_file2)
    os.rmdir(test_dir2)
    os.rmdir(test_dir)


def test_init():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.base_path == test_dir


def test_get():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    file = filesystem_wrapper.get(test_file)
    assert file == b'test1'

    file = filesystem_wrapper.get(test_file2)
    assert file == b'test2 long'


def test_get_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(FileNotFoundError):
        filesystem_wrapper.get('not_found')


def test_get_directory():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.get(test_dir2)


def test_get_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(FileNotFoundError):
        filesystem_wrapper.get(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_path'))


def test_exists():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.exists(test_file)
    assert filesystem_wrapper.exists(test_file2)
    assert filesystem_wrapper.exists(test_dir)
    assert filesystem_wrapper.exists(test_dir2)


def test_exists_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.exists('not_found')


def test_exists_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.exists(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_path'))


def test_list():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert set(filesystem_wrapper.list(test_dir)) == set(['test_file', 'test_dir'])
    assert filesystem_wrapper.list(test_dir2) == ['test_file']


def test_list_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.list('not_found')


def test_list_not_directory():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(NotADirectoryError):
        filesystem_wrapper.list(test_file)


def test_list_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.list(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_path'))


def test_list_recursive():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert set(filesystem_wrapper.list(test_dir, recursive=True)) == \
        set(['/tmp/modyn/test_dir/test_file', '/tmp/modyn/test_dir/test_dir/test_file'])


def test_list_recursive_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.list('not_found', recursive=True)


def test_list_recursive_not_directory():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(NotADirectoryError):
        filesystem_wrapper.list(test_file, recursive=True)


def test_list_recursive_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.list(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_path'), recursive=True)


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


def test_isdir_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.isdir(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_path'))


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


def test_isfile_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert not filesystem_wrapper.isfile(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_path'))


def test_get_size():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.get_size(test_file) == 5
    assert filesystem_wrapper.get_size(test_file2) == 10


def test_get_size_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_size('not_found')


def test_get_size_not_file():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.get_size(test_dir)


def test_get_size_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_size(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_path'))


def test_get_modified():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.get_modified(test_file) == TEST_FILE_MODIFIED_AT
    assert filesystem_wrapper.get_modified(test_file2) == TEST_FILE2_MODIFIED_AT


def test_get_modified_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_modified('not_found')


def test_get_modified_not_file():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.get_modified(test_dir)


def test_get_modified_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_modified(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_path'))


def test_get_created():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.get_created(test_file) == TEST_FILE_MODIFIED_AT
    assert filesystem_wrapper.get_created(test_file2) == TEST_FILE2_MODIFIED_AT


def test_get_created_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_created('not_found')


def test_get_created_not_file():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.get_created(test_dir)


def test_get_created_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_created(os.path.sep + os.path.join('tmp', 'modyn', 'not_in_base_path'))


def test_join():
    filesystem_wrapper = LocalFilesystemWrapper(test_dir)
    assert filesystem_wrapper.join('a', 'b') == 'a' + os.path.sep + 'b'
    assert filesystem_wrapper.join('a', 'b', 'c') == 'a' + os.path.sep + 'b' + os.path.sep + 'c'
    assert filesystem_wrapper.join('a', 'b', 'c', 'd') == 'a' + \
        os.path.sep + 'b' + os.path.sep + 'c' + os.path.sep + 'd'
