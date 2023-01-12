# pylint: disable=unused-argument, redefined-outer-name
import os
import pathlib
import shutil
import typing
from ctypes import c_bool
from multiprocessing import Value
from unittest.mock import patch

import pytest
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import (
    AbstractFileSystemWrapper,
)
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import (
    FilesystemWrapperType,
)
from modyn.storage.internal.new_file_watcher import NewFileWatcher

FILE_TIMESTAMP = 1600000000
TEST_DIR = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp")
TEST_FILE1 = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "test1.txt")
TEST_FILE2 = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "test2.txt")


def get_minimal_modyn_config() -> dict:
    return {
        "storage": {
            "filesystem": {"type": "LocalFilesystemWrapper", "base_path": "/tmp/modyn"},
            "database": {
                "drivername": "sqlite",
                "username": "",
                "password": "",
                "host": "",
                "port": 0,
                "database": ":memory:",
            },
            "new_file_watcher": {"interval": 1},
        }
    }


def get_invalid_modyn_config() -> dict:
    return {
        "storage": {
            "filesystem": {
                "type": "InvalidFilesystemWrapper",
                "base_path": "/tmp/modyn",
            },
            "database": {
                "drivername": "sqlite",
                "username": "",
                "password": "",
                "host": "",
                "port": 0,
                "database": ":memory:",
            },
        }
    }


def setup():
    os.makedirs(TEST_DIR, exist_ok=True)


def teardown():
    shutil.rmtree(TEST_DIR)


@pytest.fixture(autouse=True)
def session():
    with DatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_all()
        yield database.get_session()


class MockFileSystemWrapper(AbstractFileSystemWrapper):
    def __init__(self):
        super().__init__(TEST_DIR)
        self._list = [TEST_FILE1, TEST_FILE2]
        self._list_called = False

    def exists(self, path: str) -> bool:
        if path == "/notexists":
            return False
        return True

    def isdir(self, path: str) -> bool:
        if path in (TEST_FILE1, TEST_FILE2):
            return False
        if path == TEST_DIR:
            return True
        return False

    def isfile(self, path: str) -> bool:
        if path in (TEST_FILE1, TEST_FILE2):
            return True
        return False

    def list(self, path: str, recursive: bool = False) -> list[str]:  # pylint: disable=unused-argument
        self._list_called = True
        return self._list

    def join(self, *paths: str) -> str:
        return "/".join(paths)

    def get_modified(self, path: str) -> int:  # pylint: disable=unused-argument
        return FILE_TIMESTAMP

    def get_created(self, path: str) -> int:  # pylint: disable=unused-argument
        return FILE_TIMESTAMP

    def get(self, path: str) -> typing.BinaryIO:  # pylint: disable=unused-argument
        return typing.BinaryIO()

    def get_size(self, path: str) -> int:  # pylint: disable=unused-argument
        return 2

    def get_list_called(self) -> bool:
        return self._list_called


class MockFileWrapper:
    def get_number_of_samples(self) -> int:  # pylint: disable=unused-argument
        return 2


class MockDataset:
    def __init__(self):
        self.filesystem_wrapper_type = "mock"
        self.base_path = TEST_DIR


class MockFile:
    def __init__(self):
        self.path = TEST_FILE1
        self.timestamp = FILE_TIMESTAMP


class MockQuery:
    def __init__(self):
        self._all = [MockFile()]

    def all(self) -> list[MockFile]:
        return self._all


@patch.object(NewFileWatcher, "_seek_dataset", return_value=None)
def test_seek(test__seek_dataset, session) -> None:  # noqa: E501
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), should_stop)
    session.add(
        Dataset(
            name="test",
            description="test description",
            filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
            file_wrapper_type=FileWrapperType.WebdatasetFileWrapper,
            base_path=TEST_DIR,
        )
    )
    session.commit()
    with patch.object(DatabaseConnection, "get_session") as get_session_mock:
        get_session_mock.return_value = session
        new_file_watcher._seek(FILE_TIMESTAMP - 1)
        assert test__seek_dataset.called


@patch.object(NewFileWatcher, "_update_files_in_directory", return_value=None)
def test_seek_dataset(test__update_files_in_directory, session) -> None:  # noqa: E501
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), should_stop)
    session.add(
        Dataset(
            name="test",
            description="test description",
            filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
            file_wrapper_type=FileWrapperType.WebdatasetFileWrapper,
            base_path=TEST_DIR,
        )
    )
    session.commit()
    dataset = session.query(Dataset).first()
    new_file_watcher._seek_dataset(session, dataset, FILE_TIMESTAMP - 1)
    assert test__update_files_in_directory.called


@patch.object(NewFileWatcher, "_update_files_in_directory", return_value=None)
def test_seek_path_not_exists(test__update_files_in_directory, session) -> None:  # noqa: E501
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), should_stop)
    session.add(
        Dataset(
            name="test",
            description="test description",
            filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
            file_wrapper_type=FileWrapperType.WebdatasetFileWrapper,
            base_path="/notexists",
        )
    )
    session.commit()
    with patch.object(DatabaseConnection, "get_session") as get_session_mock:
        get_session_mock.return_value = session
        new_file_watcher._seek(FILE_TIMESTAMP - 1)
        assert not test__update_files_in_directory.called


@patch.object(NewFileWatcher, "_update_files_in_directory", return_value=None)
@patch(
    "modyn.storage.internal.new_file_watcher.get_filesystem_wrapper",
    return_value=MockFileSystemWrapper(),
)
def test_seek_path_not_dir(test_get_filesystem_wrapper, test__update_files_in_directory, session):  # noqa: E501
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), should_stop)
    session.add(
        Dataset(
            name="test",
            description="test description",
            filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
            file_wrapper_type=FileWrapperType.WebdatasetFileWrapper,
            base_path=TEST_FILE1,
        )
    )
    session.commit()
    with patch.object(DatabaseConnection, "get_session") as get_session_mock:
        get_session_mock.return_value = session
        new_file_watcher._seek(FILE_TIMESTAMP - 1)
        assert not test__update_files_in_directory.called


@patch.object(NewFileWatcher, "_update_files_in_directory", return_value=None)
@patch(
    "modyn.storage.internal.new_file_watcher.get_filesystem_wrapper",
    return_value=MockFileSystemWrapper(),
)
def test_seek_no_datasets(test_get_filesystem_wrapper, test__update_files_in_directory, session) -> None:  # noqa: E501
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), should_stop)
    with patch.object(DatabaseConnection, "get_session") as get_session_mock:
        get_session_mock.return_value = session
        new_file_watcher._seek(FILE_TIMESTAMP - 1)
        assert not test__update_files_in_directory.called


@patch(
    "modyn.storage.internal.new_file_watcher.get_file_wrapper",
    return_value=MockFileWrapper(),
)
@patch(
    "modyn.storage.internal.new_file_watcher.get_filesystem_wrapper",
    return_value=MockFileSystemWrapper(),
)
def test_update_files_in_directory(test_get_file_wrapper, test_get_filesystem_wrapper, session) -> None:  # noqa: E501
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), should_stop)
    dataset = Dataset(
        name="test",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_DIR,
        file_wrapper_config='{"file_extension": ".txt"}',
    )
    session.add(dataset)
    session.commit()
    new_file_watcher._update_files_in_directory(
        filesystem_wrapper=MockFileSystemWrapper(),
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        path=TEST_DIR,
        timestamp=FILE_TIMESTAMP - 1,
        session=session,
        dataset=dataset,
    )

    result = session.query(File).all()
    assert result is not None
    assert len(result) == 2
    assert result[0].path == TEST_FILE1
    assert result[0].created_at == FILE_TIMESTAMP
    assert result[0].number_of_samples == 2
    assert result[0].dataset_id == 1

    result = session.query(Sample).all()
    assert result is not None
    assert len(result) == 4
    assert result[0].file_id == 1

    new_file_watcher._update_files_in_directory(
        filesystem_wrapper=MockFileSystemWrapper(),
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        path=TEST_DIR,
        timestamp=FILE_TIMESTAMP - 1,
        session=session,
        dataset=dataset,
    )

    result = session.query(File).all()
    assert result is not None
    assert len(result) == 2
    assert result[0].path == TEST_FILE1
    assert result[0].created_at == FILE_TIMESTAMP
    assert result[0].number_of_samples == 2
    assert result[0].dataset_id == 1

    result = session.query(Sample).all()
    assert result is not None
    assert len(result) == 4
    assert result[0].file_id == 1


def test_update_files_in_directory_not_exists(session) -> None:
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), should_stop)
    mock_file_system_wrapper = MockFileSystemWrapper()
    new_file_watcher._update_files_in_directory(
        filesystem_wrapper=mock_file_system_wrapper,
        file_wrapper_type=MockFileWrapper(),
        path="/notexists",
        timestamp=FILE_TIMESTAMP - 1,
        session=session,
        dataset=MockDataset(),
    )
    assert not mock_file_system_wrapper.get_list_called()


@patch.object(NewFileWatcher, "_seek", return_value=None)
@patch("modyn.storage.internal.new_file_watcher.current_time_millis", return_value=-2)
def test_run(mock_seek, mock_time) -> None:  # pylint: disable=unused-argument
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), should_stop)
    new_file_watcher.run()
    assert new_file_watcher._seek.called
