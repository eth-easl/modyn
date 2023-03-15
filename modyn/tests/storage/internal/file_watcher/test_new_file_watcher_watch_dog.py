# pylint: disable=unused-argument, redefined-outer-name
import os
import pathlib
import shutil
import typing
from ctypes import c_bool
from multiprocessing import Process, Value
from unittest.mock import patch

import pytest
from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.file_watcher.new_file_watcher_watch_dog import NewFileWatcherWatchDog
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType

TEST_DATABASE = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "test.db")
TEST_DIR = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp")
TEST_FILE1 = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "test1.txt")


def get_minimal_modyn_config() -> dict:
    return {
        "storage": {
            "database": {
                "drivername": "sqlite",
                "username": "",
                "password": "",
                "host": "",
                "port": 0,
                "database": TEST_DATABASE,
            },
        }
    }


def get_invalid_modyn_config() -> dict:
    return {
        "storage": {
            "database": {
                "drivername": "sqlite",
                "username": "",
                "password": "",
                "host": "",
                "port": 0,
                "database": TEST_DATABASE,
            },
        }
    }


def setup():
    os.makedirs(TEST_DIR, exist_ok=True)
    with open(TEST_FILE1, "w", encoding="utf-8") as file:
        file.write("test")


def teardown():
    shutil.rmtree(TEST_DIR)


@pytest.fixture(autouse=True)
def session():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        yield database.session
        database.session.query(Dataset).delete()
        database.session.query(File).delete()
        database.session.query(Sample).delete()
        database.session.commit()


class MockProcess(Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running = Value(c_bool, True)

    def is_alive(self):
        return self._running.value

    def terminate(self):
        self._running.value = False

    def join(self, timeout: typing.Optional[float] = ...) -> None:
        pass


@patch("modyn.storage.internal.file_watcher.new_file_watcher_watch_dog.Process", return_value=MockProcess())
def test_start_file_watcher(mock_process, session):
    should_stop = Value(c_bool, False)
    new_file_watcher_watch_dog = NewFileWatcherWatchDog(get_minimal_modyn_config(), should_stop)
    new_file_watcher_watch_dog._start_file_watcher_process(1)

    assert new_file_watcher_watch_dog._file_watcher_processes[1][0] is not None


def test_stop_file_watcher_process(session):
    should_stop = Value(c_bool, False)
    new_file_watcher_watch_dog = NewFileWatcherWatchDog(get_minimal_modyn_config(), should_stop)

    mock_process = MockProcess()

    should_stop = Value(c_bool, False)

    new_file_watcher_watch_dog._file_watcher_processes[1] = (mock_process, should_stop, 0)

    new_file_watcher_watch_dog._stop_file_watcher_process(1)

    assert not mock_process.is_alive()
    assert should_stop.value


def test_watch_file_watcher_processes_dataset_not_in_database(session):
    should_stop = Value(c_bool, False)
    new_file_watcher_watch_dog = NewFileWatcherWatchDog(get_minimal_modyn_config(), should_stop)

    mock_process = MockProcess()

    should_stop = Value(c_bool, False)

    new_file_watcher_watch_dog._file_watcher_processes[1] = (mock_process, should_stop, 0)

    new_file_watcher_watch_dog._watch_file_watcher_processes()

    assert not mock_process.is_alive()
    assert should_stop.value


@patch("modyn.storage.internal.file_watcher.new_file_watcher_watch_dog.Process", return_value=MockProcess())
def test_watch_file_watcher_processes_dataset_not_in_dataset_ids_in_file_watcher_processes(mock_process, session):
    dataset = Dataset(
        name="test1",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path="/notexists",
        file_watcher_interval=0.1,
        last_timestamp=0,
    )
    session.add(dataset)
    session.commit()

    should_stop = Value(c_bool, False)

    new_file_watcher_watch_dog = NewFileWatcherWatchDog(get_minimal_modyn_config(), should_stop)

    new_file_watcher_watch_dog._watch_file_watcher_processes()

    assert dataset.dataset_id in new_file_watcher_watch_dog._file_watcher_processes
    assert new_file_watcher_watch_dog._file_watcher_processes[dataset.dataset_id][0] is not None
    assert new_file_watcher_watch_dog._file_watcher_processes[dataset.dataset_id][0].is_alive()
    assert not new_file_watcher_watch_dog._file_watcher_processes[dataset.dataset_id][1].value


@patch("modyn.storage.internal.file_watcher.new_file_watcher_watch_dog.Process", return_value=MockProcess())
def test_watch_file_watcher_processes_dataset_in_dataset_ids_in_file_watcher_processes_not_alive(mock_process, session):
    dataset = Dataset(
        name="test1",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path="/notexists",
        file_watcher_interval=0.1,
        last_timestamp=0,
    )
    session.add(dataset)
    session.commit()

    should_stop = Value(c_bool, False)

    new_file_watcher_watch_dog = NewFileWatcherWatchDog(get_minimal_modyn_config(), should_stop)

    new_file_watcher_watch_dog._file_watcher_processes[dataset.dataset_id] = (mock_process, should_stop, 0)

    mock_process.is_alive.return_value = False

    new_file_watcher_watch_dog._watch_file_watcher_processes()

    assert dataset.dataset_id in new_file_watcher_watch_dog._file_watcher_processes
    assert new_file_watcher_watch_dog._file_watcher_processes[dataset.dataset_id][0] is not None
    assert new_file_watcher_watch_dog._file_watcher_processes[dataset.dataset_id][0].is_alive()
    assert not new_file_watcher_watch_dog._file_watcher_processes[dataset.dataset_id][1].value
