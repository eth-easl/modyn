import os
import pytest
import pathlib

from modyn.storage.internal.filesystem_wrapper.local_filesystem_wrapper import LocalFilesystemWrapper

TEST_DIR = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "modyn" / "test_dir")
TEST_FILE = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "modyn" / "test_dir" / "test_file")
TEST_FILE_MODIFIED_AT = None
TEST_DIR2 = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "modyn" / "test_dir" / "test_dir")
TEST_FILE2 = str(
    pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "modyn" / "test_dir" / "test_dir" / "test_file2"
)
TEST_FILE2_MODIFIED_AT = None


def setup():
    os.makedirs(TEST_DIR, exist_ok=True)

    with open(TEST_FILE, "w", encoding="utf8") as file:
        file.write("test1")

    global TEST_FILE_MODIFIED_AT  #  pylint: disable=global-statement # noqa: E262
    TEST_FILE_MODIFIED_AT = int(os.path.getmtime(TEST_FILE) * 1000)

    os.makedirs(TEST_DIR2, exist_ok=True)

    with open(TEST_FILE2, "w", encoding="utf8") as file:
        file.write("test2 long")

    global TEST_FILE2_MODIFIED_AT  #  pylint: disable=global-statement # noqa: E262
    TEST_FILE2_MODIFIED_AT = int(os.path.getmtime(TEST_FILE2) * 1000)


def teardown():
    os.remove(TEST_FILE)
    os.remove(TEST_FILE2)
    os.rmdir(TEST_DIR2)
    os.rmdir(TEST_DIR)


def test_init():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert filesystem_wrapper.base_path == TEST_DIR


def test_get():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    file = filesystem_wrapper.get(TEST_FILE)
    assert file == b"test1"

    file = filesystem_wrapper.get(TEST_FILE2)
    assert file == b"test2 long"


def test_get_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.get("not_found")


def test_get_directory():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.get(TEST_DIR2)


def test_get_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.get(os.path.sep + os.path.join("tmp", "modyn", "not_in_base_path"))


def test_exists():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert filesystem_wrapper.exists(TEST_FILE)
    assert filesystem_wrapper.exists(TEST_FILE2)
    assert filesystem_wrapper.exists(TEST_DIR)
    assert filesystem_wrapper.exists(TEST_DIR2)


def test_exists_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert not filesystem_wrapper.exists("not_found")


def test_exists_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert not filesystem_wrapper.exists(os.path.sep + os.path.join("tmp", "modyn", "not_in_base_path"))


def test_list():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert set(filesystem_wrapper.list(TEST_DIR)) == set(["test_file", "test_dir"])
    assert filesystem_wrapper.list(TEST_DIR2) == ["test_file2"]


def test_list_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.list("not_found")


def test_list_not_directory():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(NotADirectoryError):
        filesystem_wrapper.list(TEST_FILE)


def test_list_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.list(os.path.sep + os.path.join("tmp", "modyn", "not_in_base_path"))


def test_list_recursive():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert set(filesystem_wrapper.list(TEST_DIR, recursive=True)) == set([TEST_FILE, TEST_FILE2])


def test_list_recursive_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.list("not_found", recursive=True)


def test_list_recursive_not_directory():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(NotADirectoryError):
        filesystem_wrapper.list(TEST_FILE, recursive=True)


def test_list_recursive_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.list(os.path.sep + os.path.join("tmp", "modyn", "not_in_base_path"), recursive=True)


def test_isdir():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert filesystem_wrapper.isdir(TEST_DIR)
    assert filesystem_wrapper.isdir(TEST_DIR2)


def test_isdir_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert not filesystem_wrapper.isdir("not_found")


def test_isdir_not_directory():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert not filesystem_wrapper.isdir(TEST_FILE)


def test_isdir_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert not filesystem_wrapper.isdir(os.path.sep + os.path.join("tmp", "modyn", "not_in_base_path"))


def test_isfile():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert filesystem_wrapper.isfile(TEST_FILE)
    assert filesystem_wrapper.isfile(TEST_FILE2)


def test_isfile_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert not filesystem_wrapper.isfile("not_found")


def test_isfile_not_directory():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert not filesystem_wrapper.isfile(TEST_DIR)


def test_isfile_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert not filesystem_wrapper.isfile(os.path.sep + os.path.join("tmp", "modyn", "not_in_base_path"))


def test_get_size():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert filesystem_wrapper.get_size(TEST_FILE) == 5
    assert filesystem_wrapper.get_size(TEST_FILE2) == 10


def test_get_size_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_size("not_found")


def test_get_size_not_file():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.get_size(TEST_DIR)


def test_get_size_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_size(os.path.sep + os.path.join("tmp", "modyn", "not_in_base_path"))


def test_get_modified():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert filesystem_wrapper.get_modified(TEST_FILE) == TEST_FILE_MODIFIED_AT
    assert filesystem_wrapper.get_modified(TEST_FILE2) == TEST_FILE2_MODIFIED_AT


def test_get_modified_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_modified("not_found")


def test_get_modified_not_file():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.get_modified(TEST_DIR)


def test_get_modified_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_modified(os.path.sep + os.path.join("tmp", "modyn", "not_in_base_path"))


def test_get_created():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert filesystem_wrapper.get_created(TEST_FILE) == TEST_FILE_MODIFIED_AT
    assert filesystem_wrapper.get_created(TEST_FILE2) == TEST_FILE2_MODIFIED_AT


def test_get_created_not_found():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_created("not_found")


def test_get_created_not_file():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(IsADirectoryError):
        filesystem_wrapper.get_created(TEST_DIR)


def test_get_created_not_in_base_path():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    with pytest.raises(ValueError):
        filesystem_wrapper.get_created(os.path.sep + os.path.join("tmp", "modyn", "not_in_base_path"))


def test_join():
    filesystem_wrapper = LocalFilesystemWrapper(TEST_DIR)
    assert filesystem_wrapper.join("a", "b") == "a" + os.path.sep + "b"
    assert filesystem_wrapper.join("a", "b", "c") == "a" + os.path.sep + "b" + os.path.sep + "c"
    assert (
        filesystem_wrapper.join("a", "b", "c", "d") == "a" + os.path.sep + "b" + os.path.sep + "c" + os.path.sep + "d"
    )
