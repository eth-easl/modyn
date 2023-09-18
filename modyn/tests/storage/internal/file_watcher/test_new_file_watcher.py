# pylint: disable=unused-argument, redefined-outer-name
import os
import pathlib
import shutil
import time
import typing
from ctypes import c_bool
from multiprocessing import Process, Value
from unittest.mock import patch

import pytest
from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.file_watcher.new_file_watcher import NewFileWatcher, run_new_file_watcher
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType

FILE_TIMESTAMP = 1600000000
TEST_DIR = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp")
TEST_FILE1 = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "test1.txt")
TEST_FILE2 = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "test2.txt")
TEST_FILE_WRONG_SUFFIX = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "test1.csv")
TEST_DATABASE = str(pathlib.Path(os.path.abspath(__file__)).parent / "tmp" / "test.db")


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
            "insertion_threads": 8,
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
    with open(TEST_FILE2, "w", encoding="utf-8") as file:
        file.write("test")


def teardown():
    shutil.rmtree(TEST_DIR)


@pytest.fixture(autouse=True)
def storage_database_connection():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        yield database
        database.session.query(Dataset).delete()
        database.session.query(File).delete()
        database.session.query(Sample).delete()
        database.session.commit()


class MockFileSystemWrapper(AbstractFileSystemWrapper):
    def __init__(self):
        super().__init__(TEST_DIR)
        self._list = [TEST_FILE1, TEST_FILE2, TEST_FILE_WRONG_SUFFIX]
        self._list_called = False

    def exists(self, path: str) -> bool:
        if path == "/notexists":
            return False
        return True

    def isdir(self, path: str) -> bool:
        if path in (TEST_FILE1, TEST_FILE2, TEST_FILE_WRONG_SUFFIX):
            return False
        if path == TEST_DIR:
            return True
        return False

    def isfile(self, path: str) -> bool:
        if path in (TEST_FILE1, TEST_FILE2, TEST_FILE_WRONG_SUFFIX):
            return True
        return False

    def list(self, path: str, recursive: bool = False) -> list[str]:
        self._list_called = True
        return self._list

    def join(self, *paths: str) -> str:
        return "/".join(paths)

    def get_modified(self, path: str) -> int:
        return FILE_TIMESTAMP

    def get_created(self, path: str) -> int:
        return FILE_TIMESTAMP

    def _get(self, path: str) -> typing.BinaryIO:
        return typing.BinaryIO()

    def get_size(self, path: str) -> int:
        return 2

    def get_list_called(self) -> bool:
        return self._list_called

    def delete(self, path: str) -> None:
        return


class MockFileWrapper:
    def get_number_of_samples(self) -> int:
        return 2

    def get_label(self, index: int) -> bytes:
        return b"test"

    def get_all_labels(self) -> list[bytes]:
        return [b"test", b"test"]


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
def test_seek(test__seek_dataset, storage_database_connection) -> None:  # noqa: E501
    session = storage_database_connection.session
    dataset = Dataset(
        name="test1",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_DIR,
        last_timestamp=FILE_TIMESTAMP - 1,
        file_watcher_interval=0.1,
    )
    session.add(dataset)
    session.commit()

    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop)

    session.add(
        File(dataset=dataset, path="/tmp/modyn/test", created_at=0, updated_at=FILE_TIMESTAMP + 10, number_of_samples=1)
    )
    session.commit()

    new_file_watcher._seek(storage_database_connection, dataset)
    assert test__seek_dataset.called
    assert session.query(Dataset).first().last_timestamp == FILE_TIMESTAMP + 10


@patch.object(NewFileWatcher, "_update_files_in_directory", return_value=None)
def test_seek_dataset(test__update_files_in_directory, storage_database_connection) -> None:  # noqa: E501
    should_stop = Value(c_bool, False)

    session = storage_database_connection.session

    session.add(
        Dataset(
            name="test2",
            description="test description",
            filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
            file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
            base_path=TEST_DIR,
            last_timestamp=FILE_TIMESTAMP - 1,
            file_watcher_interval=0.1,
        )
    )
    session.commit()
    dataset = session.query(Dataset).first()

    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop)

    new_file_watcher._seek_dataset(session, dataset)
    assert test__update_files_in_directory.called


def test_seek_dataset_deleted(storage_database_connection) -> None:  # noqa: E501
    should_stop = Value(c_bool, False)

    session = storage_database_connection.session

    session.add(
        Dataset(
            name="test2",
            description="test description",
            filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
            file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
            base_path=TEST_DIR,
            file_wrapper_config='{"file_extension": ".txt"}',
            last_timestamp=FILE_TIMESTAMP - 1,
            file_watcher_interval=0.1,
        )
    )
    session.commit()

    dataset = session.query(Dataset).first()
    session.add(
        File(dataset=dataset, path="/tmp/modyn/test", created_at=0, updated_at=FILE_TIMESTAMP + 10, number_of_samples=1)
    )
    session.commit()

    process = Process(target=NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop).run)
    process.start()

    start = time.time()

    time.sleep(1)

    session.delete(dataset)
    session.commit()

    while time.time() - start < 5:
        if not process.is_alive():
            break
        time.sleep(0.1)

    assert not process.is_alive()


@patch.object(NewFileWatcher, "_update_files_in_directory", return_value=None)
@patch(
    "modyn.storage.internal.file_watcher.new_file_watcher.get_filesystem_wrapper", return_value=MockFileSystemWrapper()
)
def test_seek_path_not_exists(
    test_get_filesystem_wrapper, test__update_files_in_directory, storage_database_connection
) -> None:  # noqa: E501
    session = storage_database_connection.session
    dataset = Dataset(
        name="test1",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path="/notexists",
        last_timestamp=FILE_TIMESTAMP - 1,
        file_watcher_interval=0.1,
    )
    session.add(dataset)
    session.commit()
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop)
    session.add(
        File(dataset=dataset, path="/tmp/modyn/test", created_at=0, updated_at=FILE_TIMESTAMP + 10, number_of_samples=1)
    )
    session.commit()

    new_file_watcher._seek(storage_database_connection, dataset)
    assert not test__update_files_in_directory.called
    assert session.query(Dataset).first().last_timestamp == FILE_TIMESTAMP + 10


@patch.object(NewFileWatcher, "_update_files_in_directory", return_value=None)
@patch(
    "modyn.storage.internal.file_watcher.new_file_watcher.get_filesystem_wrapper", return_value=MockFileSystemWrapper()
)
def test_seek_path_not_directory(
    test_get_filesystem_wrapper, test__update_files_in_directory, storage_database_connection
) -> None:  # noqa: E501
    session = storage_database_connection.session
    dataset = Dataset(
        name="test1",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_FILE1,
        last_timestamp=FILE_TIMESTAMP - 1,
        file_watcher_interval=0.1,
    )
    session.add(dataset)
    session.commit()
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop)
    session.add(
        File(dataset=dataset, path="/tmp/modyn/test", created_at=0, updated_at=FILE_TIMESTAMP + 10, number_of_samples=1)
    )
    session.commit()

    new_file_watcher._seek(storage_database_connection, dataset)
    assert not test__update_files_in_directory.called
    assert session.query(Dataset).first().last_timestamp == FILE_TIMESTAMP + 10


@patch.object(NewFileWatcher, "_update_files_in_directory", return_value=None)
@patch(
    "modyn.storage.internal.file_watcher.new_file_watcher.get_filesystem_wrapper", return_value=MockFileSystemWrapper()
)
def test_seek_no_datasets(
    test_get_filesystem_wrapper, test__update_files_in_directory, storage_database_connection
) -> None:  # noqa: E501
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), 1, should_stop)

    new_file_watcher._seek(storage_database_connection, None)
    assert not test__update_files_in_directory.called


@patch("modyn.storage.internal.file_watcher.new_file_watcher.get_file_wrapper", return_value=MockFileWrapper())
@patch(
    "modyn.storage.internal.file_watcher.new_file_watcher.get_filesystem_wrapper", return_value=MockFileSystemWrapper()
)
def test_update_files_in_directory(
    test_get_file_wrapper, test_get_filesystem_wrapper, storage_database_connection
) -> None:  # noqa: E501
    session = storage_database_connection.session
    dataset = Dataset(
        name="test5",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_DIR,
        file_wrapper_config='{"file_extension": ".txt"}',
        last_timestamp=FILE_TIMESTAMP - 1,
        file_watcher_interval=0.1,
    )
    session.add(dataset)
    session.commit()

    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop)

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


@patch("modyn.storage.internal.file_watcher.new_file_watcher.get_file_wrapper", return_value=MockFileWrapper())
@patch(
    "modyn.storage.internal.file_watcher.new_file_watcher.get_filesystem_wrapper", return_value=MockFileSystemWrapper()
)
def test_update_files_in_directory_mt_disabled(
    test_get_file_wrapper, test_get_filesystem_wrapper, storage_database_connection
) -> None:  # noqa: E501
    session = storage_database_connection.session
    dataset = Dataset(
        name="test5",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_DIR,
        file_wrapper_config='{"file_extension": ".txt"}',
        last_timestamp=FILE_TIMESTAMP - 1,
        file_watcher_interval=0.1,
    )
    session.add(dataset)
    session.commit()

    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop)
    new_file_watcher._disable_mt = True

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


@patch("modyn.storage.internal.file_watcher.new_file_watcher.get_file_wrapper", return_value=MockFileWrapper())
@patch(
    "modyn.storage.internal.file_watcher.new_file_watcher.get_filesystem_wrapper", return_value=MockFileSystemWrapper()
)
def test_handle_file_paths_presupplied_config(
    test_get_file_wrapper, test_get_filesystem_wrapper, storage_database_connection
) -> None:  # noqa: E501
    session = storage_database_connection.session
    dataset = Dataset(
        name="test_handle_file_paths",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_DIR,
        file_wrapper_config='{"file_extension": ".txt"}',
        last_timestamp=FILE_TIMESTAMP - 1,
        file_watcher_interval=0.1,
    )

    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop)

    session.add(dataset)
    session.commit()

    file_paths = MockFileSystemWrapper().list(TEST_DIR, recursive=True)
    new_file_watcher._handle_file_paths(
        -1,
        1234,
        False,
        False,
        file_paths,
        get_minimal_modyn_config(),
        ".txt",
        MockFileSystemWrapper(),
        "fw",
        FILE_TIMESTAMP - 1,
        "test_handle_file_paths",
        1,
        session,
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

    new_file_watcher._handle_file_paths(
        -1,
        1234,
        False,
        False,
        file_paths,
        get_minimal_modyn_config(),
        ".txt",
        MockFileSystemWrapper(),
        "fw",
        FILE_TIMESTAMP - 1,
        "test_handle_file_paths",
        1,
        session,
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


@patch("modyn.storage.internal.file_watcher.new_file_watcher.get_file_wrapper", return_value=MockFileWrapper())
@patch(
    "modyn.storage.internal.file_watcher.new_file_watcher.get_filesystem_wrapper", return_value=MockFileSystemWrapper()
)
def test_handle_file_paths_no_presupplied_config(
    test_get_file_wrapper, test_get_filesystem_wrapper, storage_database_connection
) -> None:  # noqa: E501
    session = storage_database_connection.session
    dataset = Dataset(
        name="test_handle_file_paths",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_DIR,
        file_wrapper_config='{"file_extension": ".txt"}',
        last_timestamp=FILE_TIMESTAMP - 1,
        file_watcher_interval=0.1,
    )

    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop)

    session.add(dataset)
    session.commit()

    file_paths = MockFileSystemWrapper().list(TEST_DIR, recursive=True)
    new_file_watcher._handle_file_paths(
        -1,
        1234,
        False,
        False,
        file_paths,
        get_minimal_modyn_config(),
        ".txt",
        MockFileSystemWrapper(),
        "fw",
        FILE_TIMESTAMP - 1,
        "test_handle_file_paths",
        1,
        None,
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

    new_file_watcher._handle_file_paths(
        -1,
        1234,
        False,
        False,
        file_paths,
        get_minimal_modyn_config(),
        ".txt",
        MockFileSystemWrapper(),
        "fw",
        FILE_TIMESTAMP - 1,
        "test_handle_file_paths",
        1,
        None,
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


@patch("modyn.storage.internal.file_watcher.new_file_watcher.get_file_wrapper", return_value=MockFileWrapper())
@patch(
    "modyn.storage.internal.file_watcher.new_file_watcher.get_filesystem_wrapper", return_value=MockFileSystemWrapper()
)
def test_update_files_in_directory_ignore_last_timestamp(
    test_get_file_wrapper, test_get_filesystem_wrapper, storage_database_connection
) -> None:  # noqa: E501
    session = storage_database_connection.session
    dataset = Dataset(
        name="test6",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_DIR,
        file_wrapper_config='{"file_extension": ".txt"}',
        last_timestamp=FILE_TIMESTAMP - 1,
        ignore_last_timestamp=True,
        file_watcher_interval=0.1,
    )
    session.add(dataset)
    session.commit()

    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop)

    new_file_watcher._update_files_in_directory(
        filesystem_wrapper=MockFileSystemWrapper(),
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        path=TEST_DIR,
        timestamp=FILE_TIMESTAMP + 10,
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


def test_update_files_in_directory_not_exists(storage_database_connection) -> None:
    session = storage_database_connection.session
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), 1, should_stop)
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
def test_run(mock_seek, storage_database_connection) -> None:
    session = storage_database_connection.session
    dataset = Dataset(
        name="test7",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_DIR,
        file_wrapper_config='{"file_extension": ".txt"}',
        last_timestamp=-1,
    )
    session.add(dataset)
    session.commit()

    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), dataset.dataset_id, should_stop)
    watcher_process = Process(target=new_file_watcher.run, args=())
    watcher_process.start()
    should_stop.value = True  # type: ignore
    watcher_process.join()
    #  If we get here, the process has stopped


def test_get_datasets(storage_database_connection):
    session = storage_database_connection.session
    should_stop = Value(c_bool, False)
    new_file_watcher = NewFileWatcher(get_minimal_modyn_config(), 1, should_stop)
    datasets = new_file_watcher._get_datasets(session)
    assert len(datasets) == 0

    dataset = Dataset(
        name="test_get_datasets",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_DIR,
        last_timestamp=FILE_TIMESTAMP - 1,
        file_wrapper_config='{"file_extension": ".txt"}',
        file_watcher_interval=0.1,
        ignore_last_timestamp=True,
    )
    session.add(dataset)
    session.commit()

    datasets: list[Dataset] = new_file_watcher._get_datasets(session)
    assert len(datasets) == 1
    assert datasets[0].name == "test_get_datasets"


def test_run_new_file_watcher(storage_database_connection):
    session = storage_database_connection.session
    should_stop = Value(c_bool, False)

    dataset = Dataset(
        name="test8",
        description="test description",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        base_path=TEST_DIR,
        last_timestamp=FILE_TIMESTAMP - 1,
        file_wrapper_config='{"file_extension": ".txt"}',
        file_watcher_interval=0.1,
        ignore_last_timestamp=True,
    )
    session.add(dataset)
    session.commit()

    Process(target=run_new_file_watcher, args=(get_minimal_modyn_config(), dataset.dataset_id, should_stop)).start()

    time.sleep(2)  # If this test fails, try increasing this number
    should_stop.value = True  # type: ignore

    result = session.query(File).filter(File.path == TEST_FILE1).all()
    assert result is not None
    assert len(result) == 1
    assert result[0].path == TEST_FILE1
    assert result[0].number_of_samples == 1
    assert result[0].dataset_id == 1
