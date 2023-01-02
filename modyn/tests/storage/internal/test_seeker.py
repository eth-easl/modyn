import datetime
import typing
from unittest.mock import patch
import pytest

from modyn.storage.internal.seeker import Seeker
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType
from modyn.storage.internal.file_wrapper.mnist_webdataset_file_wrapper import MNISTWebdatasetFileWrapper
from modyn.storage.internal.filesystem_wrapper.local_filesystem_wrapper import LocalFilesystemWrapper


def get_minimal_modyn_config() -> dict:
    return {
        'storage': {
            'filesystem': {
                'type': 'LocalFilesystemWrapper',
                'base_path': '/tmp/modyn'
            },
            'database': {
                'drivername': 'sqlite',
                'username': '',
                'password': '',
                'host': '',
                'port': 0,
                'database': ':memory:'
            },
            'seeker': {
                'interval': 1
            }
        }
    }


def get_invalid_modyn_config() -> dict:
    return {
        'storage': {
            'filesystem': {
                'type': 'InvalidFilesystemWrapper',
                'base_path': '/tmp/modyn'
            },
            'database': {
                'drivername': 'sqlite',
                'username': '',
                'password': '',
                'host': '',
                'port': 0,
                'database': ':memory:'
            }
        }
    }


@pytest.fixture(autouse=True)
def session():
    with DatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_all()
        yield database.get_session()


class MockFileSystemWrapper(AbstractFileSystemWrapper):
    def __init__(self):
        super().__init__('/path')
        self._list = ['/path/file1', '/path/file2']

    def exists(self, path: str) -> bool:
        if path == '/notexists':
            return False
        return True

    def isdir(self, path: str) -> bool:
        if path in ('/path/file1', '/path/file2'):
            return False
        if path == '/path':
            return True
        return False

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

    def get_size(self) -> int:  # pylint: disable=unused-argument
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


@patch.object(Seeker, '_update_files_in_directory', return_value=None)
@patch.object(Seeker, '_get_filesystem_wrapper', return_value=MockFileSystemWrapper())
def test_seek(test__get_filesystem_wrapper, test__update_files_in_directory, session) -> None:  # pylint: disable=unused-argument, redefined-outer-name # noqa: E501
    seeker = Seeker(get_minimal_modyn_config())
    session.add(
        Dataset(
            name='test',
            description='test description',
            filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
            file_wrapper_type=FileWrapperType.MNISTWebdatasetFileWrapper,
            base_path='/path'))
    session.commit()
    seeker._session = session
    seeker._seek(datetime.datetime(2020, 1, 1))
    assert test__update_files_in_directory.called


@patch.object(Seeker, '_update_files_in_directory', return_value=None)
def test_seek_path_not_exists(test__update_files_in_directory, session) -> None:  # pylint: disable=unused-argument, redefined-outer-name # noqa: E501
    seeker = Seeker(get_minimal_modyn_config())
    session.add(
        Dataset(
            name='test',
            description='test description',
            filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
            file_wrapper_type=FileWrapperType.MNISTWebdatasetFileWrapper,
            base_path='/notexists'))
    session.commit()
    seeker._session = session
    seeker._seek(datetime.datetime(2020, 1, 1))
    assert not test__update_files_in_directory.called


@patch.object(Seeker, '_update_files_in_directory', return_value=None)
@patch.object(Seeker, '_get_filesystem_wrapper', return_value=MockFileSystemWrapper())
def test_seek_path_not_dir(test__get_filesystem_wrapper, test__update_files_in_directory, session):  # pylint: disable=unused-argument, redefined-outer-name # noqa: E501
    seeker = Seeker(get_minimal_modyn_config())
    session.add(
        Dataset(
            name='test',
            description='test description',
            filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
            file_wrapper_type=FileWrapperType.MNISTWebdatasetFileWrapper,
            base_path='/path/file1'))
    session.commit()
    seeker._session = session
    seeker._seek(datetime.datetime(2020, 1, 1))
    assert not test__update_files_in_directory.called


@patch.object(Seeker, '_update_files_in_directory', return_value=None)
@patch.object(Seeker, '_get_filesystem_wrapper', return_value=MockFileSystemWrapper())
def test_seek_no_datasets(test_get_filesystem_wrapper, test__update_files_in_directory, session) -> None:  # pylint: disable=unused-argument, redefined-outer-name # noqa: E501
    seeker = Seeker(get_minimal_modyn_config())
    seeker._session = session
    seeker._seek(datetime.datetime(2020, 1, 1))
    assert not test__update_files_in_directory.called


@patch.object(Seeker, '_get_file_wrapper', return_value=MockFileWrapper())
@patch.object(Seeker, '_get_filesystem_wrapper', return_value=MockFileSystemWrapper())
def test_update_files_in_directory(test_get_file_wrapper, test_get_filesystem_wrapper, session) -> None:  # pylint: disable=unused-argument, redefined-outer-name # noqa: E501
    seeker = Seeker(get_minimal_modyn_config())
    dataset = Dataset(
        name='test',
        description='test description',
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.MNISTWebdatasetFileWrapper,
        base_path='/path')
    session.add(dataset)
    session.commit()
    seeker._update_files_in_directory(
        filesystem_wrapper=MockFileSystemWrapper(),
        file_wrapper_type=MockFileWrapper(),
        path='/path',
        timestamp=datetime.datetime(2020, 1, 1),
        session=session,
        dataset=dataset
    )

    result = session.query(File).all()
    assert result is not None
    assert len(result) == 2
    assert result[0].path == '/path/file1'
    assert result[0].created_at == datetime.datetime(2021, 1, 1)
    assert result[0].number_of_samples == 2
    assert result[0].dataset_id == 1

    result = session.query(Sample).all()
    assert result is not None
    assert len(result) == 4
    assert result[0].file_id == 1

    seeker._update_files_in_directory(
        filesystem_wrapper=MockFileSystemWrapper(),
        file_wrapper_type=MockFileWrapper(),
        path='/path',
        timestamp=datetime.datetime(2020, 1, 1),
        session=session,
        dataset=dataset
    )

    result = session.query(File).all()
    assert result is not None
    assert len(result) == 2
    assert result[0].path == '/path/file1'
    assert result[0].created_at == datetime.datetime(2021, 1, 1)
    assert result[0].number_of_samples == 2
    assert result[0].dataset_id == 1

    result = session.query(Sample).all()
    assert result is not None
    assert len(result) == 4
    assert result[0].file_id == 1


def test_update_files_in_directory_not_exists(session) -> None:  # pylint: disable=unused-argument, redefined-outer-name
    seeker = Seeker(get_minimal_modyn_config())
    with pytest.raises(ValueError):
        seeker._update_files_in_directory(
            filesystem_wrapper=MockFileSystemWrapper(),
            file_wrapper_type=MockFileWrapper(),
            path='/notexists',
            timestamp=datetime.datetime(2020, 1, 1),
            session=session,
            dataset=MockDataset()
        )


def test_get_database_session() -> None:
    seeker = Seeker(get_minimal_modyn_config())
    sess = seeker._get_database_session()
    assert sess is not None


def test_get_filesystem_wrapper() -> None:
    seeker = Seeker(get_minimal_modyn_config())
    filesystem_wrapper = seeker._get_filesystem_wrapper(FilesystemWrapperType.LocalFilesystemWrapper, '/path')
    assert filesystem_wrapper is not None
    assert isinstance(filesystem_wrapper, LocalFilesystemWrapper)


def test_get_file_wrapper() -> None:
    seeker = Seeker(get_minimal_modyn_config())
    file_wrapper = seeker._get_file_wrapper(FileWrapperType.MNISTWebdatasetFileWrapper, '/path')
    assert file_wrapper is not None
    assert isinstance(file_wrapper, MNISTWebdatasetFileWrapper)


@patch.object(Seeker, '_seek', return_value=None)
def test_run(mock_seek) -> None:  # pylint: disable=unused-argument
    seeker = Seeker(get_minimal_modyn_config())
    seeker._testing = True
    seeker.run()
    assert seeker._seek.called
