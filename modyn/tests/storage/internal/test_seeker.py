import datetime
import typing
from unittest.mock import patch
import pytest

from modyn.storage.internal.seeker import Seeker
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database import storage_database_utils

mock_config = {
    'database': {
        'type': 'sqlite',
        'path': ':memory:'
    }
}

database = DatabaseConnection(mock_config)


def setup():
    database.create_tables()

    database.get_session().add_all([
        MockDataset()
    ])

    database.get_session().commit()


class MockFileSystemWrapper(AbstractFileSystemWrapper):
    def __init__(self):
        super().__init__('/path')
        self._exists = True
        self._list = ['file1', 'file2']

    def exists(self, path: str) -> bool:
        if path == '/notexists':
            return False
        return self._exists

    def isdir(self, path: str) -> bool:
        if path in ('/path/file1', '/path/file2'):
            return False
        return True

    def isfile(self, path: str) -> bool:
        if path in ('/path/file1', '/path/file2'):
            return True
        return False

    def list(self, path: str, recursive: bool = False) -> typing.List[str]:  # pylint: disable=unused-argument
        return self._list

    def join(self, *paths: str) -> str:
        return '/'.join(paths)

    def get_modified(self, path: str) -> datetime.datetime:  # pylint: disable=unused-argument
        return datetime.datetime(2021, 1, 1)

    def get_created(self, path: str) -> datetime.datetime:  # pylint: disable=unused-argument
        return datetime.datetime(2021, 1, 1)

    def get(self, path: str) -> typing.BinaryIO:  # pylint: disable=unused-argument
        return typing.BinaryIO()

    def get_size(self, path: str) -> int:  # pylint: disable=unused-argument
        return 2


class MockFileWrapper:
    def __init__(self):
        self._timestamp = datetime.datetime(2021, 1, 1)

    def get_modified(self, path: str) -> datetime.datetime:  # pylint: disable=unused-argument
        return self._timestamp

    def get_size(self, path: str) -> int:  # pylint: disable=unused-argument
        return 2


class MockDataset:
    def __init__(self):
        self.filesystem_wrapper_type = 'mock'
        self.base_path = '/path'


class MockFile:
    def __init__(self):
        self.path = '/path/file1'
        self.timestamp = datetime.datetime(2021, 1, 1)


class MockQuery:
    def __init__(self):
        self._all = [MockFile()]

    def all(self) -> typing.List[MockFile]:
        return self._all


class MockSession:
    def __init__(self):
        self._query = MockQuery()

    def query(self, *args, **kwargs) -> MockQuery:  # pylint: disable=unused-argument
        return self._query

    def add(self, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        database.get_session().add(*args, **kwargs)


@patch.object(Seeker, '_update_files_in_directory', return_value=None)
@patch.object(storage_database_utils, 'get_filesystem_wrapper', return_value=MockFileSystemWrapper())
def test_seek() -> None:
    seeker = Seeker(mock_config)
    seeker._seek(datetime.datetime(2020, 1, 1))


@patch.object(storage_database_utils, 'get_file_wrapper', return_value=MockFileWrapper())
def test_update_files_in_directory() -> None:
    seeker = Seeker(mock_config)
    seeker._update_files_in_directory(
        filesystem_wrapper=MockFileSystemWrapper(),
        path='/path',
        timestamp=datetime.datetime(2020, 1, 1),
        session=MockSession(),
        dataset=MockDataset()
    )

    assert database.get_session().query(File).all()[0].path == '/path/file1'
    assert database.get_session().query(File).all()[0].timestamp == datetime.datetime(2021, 1, 1)
    assert database.get_session().query(File).all()[0].size == 2
    assert database.get_session().query(File).all()[0].dataset_id == 1
    assert database.get_session().query(Sample).all()[0].timestamp == datetime.datetime(2021, 1, 1)
    assert database.get_session().query(Sample).all()[0].file_id == 1


def test_update_files_in_directory_not_exists() -> None:
    seeker = Seeker(mock_config)
    with pytest.raises(ValueError):
        seeker._update_files_in_directory(
            filesystem_wrapper=MockFileSystemWrapper(),
            path='/notexists',
            timestamp=datetime.datetime(2020, 1, 1),
            session=MockSession(),
            dataset=MockDataset()
        )
