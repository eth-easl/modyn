import os
import datetime
import pickle
from unittest.mock import patch
from webdataset import WebDataset, TarWriter
import pathlib

from modyn.storage.internal.grpc.storage_grpc_server import StorageGRPCServer
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.grpc.generated.storage_pb2 import GetRequest, \
    GetNewDataSinceRequest, DatasetAvailableRequest, RegisterNewDatasetRequest  # pylint: disable=no-name-in-module
from modyn.storage.internal.file_wrapper.webdataset_file_wrapper import WebdatasetFileWrapper
from modyn.storage.internal.filesystem_wrapper.local_filesystem_wrapper import LocalFilesystemWrapper

TMP_FILE = str(pathlib.Path(os.path.abspath(__file__)).parent / 'test.tar')
TMP_FILE2 = str(pathlib.Path(os.path.abspath(__file__)).parent / 'test2.tar')
DATABASE = pathlib.Path(os.path.abspath(__file__)).parent / 'test_storage.database'


def get_minimal_modyn_config() -> dict:
    return {
        'storage': {
            'filesystem': {
                'type': 'LocalFilesystemWrapper',
                'base_path': os.path.dirname(TMP_FILE)
            },
            'database': {
                'drivername': 'sqlite',
                'username': '',
                'password': '',
                'host': '',
                'port': '0',
                'database': f'{DATABASE}'
            },
            'seeker': {
                'interval': 1
            },
            'datasets': [
                {
                    'name': 'test',
                    'base_path': os.path.dirname(TMP_FILE),
                    'filesystem_wrapper_type': LocalFilesystemWrapper,
                    'file_wrapper_type': WebdatasetFileWrapper,
                    'description': 'test',
                    'version': '0.0.1',
                }
            ]
        },
        'project': {
            'name': 'test',
            'version': '0.0.1'
        },
        'input': {
            'type': 'LOCAL',
            'path': os.path.dirname(TMP_FILE)
        },
        'odm': {
            'type': 'LOCAL'
        }
    }


def setup():

    os.makedirs(os.path.dirname(TMP_FILE), exist_ok=True)
    writer = TarWriter(TMP_FILE)
    writer.write({'__key__': '1', 'cls': [1, 2, 3], 'json': [1, 2, 3]})
    writer.write({'__key__': '2', 'cls': [1, 2, 3], 'json': [1, 2, 3]})
    writer.close()

    writer = TarWriter(TMP_FILE2)
    writer.write({'__key__': '3', 'cls': [1, 2, 3], 'json': [1, 2, 3]})
    writer.write({'__key__': '4', 'cls': [1, 2, 3], 'json': [1, 2, 3]})
    writer.close()

    with DatabaseConnection(get_minimal_modyn_config()) as database:
        now = datetime.datetime.now()
        database.create_all()

        session = database.get_session()

        dataset = Dataset(
            name='test',
            base_path=os.path.dirname(TMP_FILE),
            filesystem_wrapper_type='LocalFilesystemWrapper',
            file_wrapper_type='WebdatasetFileWrapper',
            description='test',
            version='0.0.1'
        )

        session.add(dataset)

        session.commit()

        file = File(
            path=TMP_FILE,
            dataset=dataset,
            created_at=now,
            updated_at=now,
            number_of_samples=2
        )

        session.add(file)

        file2 = File(
            path=TMP_FILE2,
            dataset=dataset,
            created_at=now,
            updated_at=now,
            number_of_samples=2
        )

        session.commit()

        sample = Sample(
            file=file,
            index=0,
            external_key='test'
        )

        session.add(sample)

        sample2 = Sample(
            file=file,
            index=1,
            external_key='test2'
        )

        session.add(sample2)

        sample3 = Sample(
            file=file2,
            index=0,
            external_key='test3'
        )

        session.add(sample3)

        sample4 = Sample(
            file=file2,
            index=1,
            external_key='test4'
        )

        session.add(sample4)

        session.commit()


def teardown():
    os.remove(DATABASE)
    os.remove(TMP_FILE)
    os.remove(TMP_FILE2)


def test_init() -> None:
    server = StorageGRPCServer(get_minimal_modyn_config())
    assert server is not None


@patch.object(WebdatasetFileWrapper, 'get_samples_from_indices', return_value=b'')
def test_get(mock_get_samples_from_indices):
    server = StorageGRPCServer(get_minimal_modyn_config())

    request = GetRequest(dataset_id='test', keys=['test', 'test3', 'test4'])

    for response in server.Get(request, None):
        assert response is not None
        assert response.chunk == b''
        mock_get_samples_from_indices.assert_called_once_with([0])


def test_get_invalid_dataset():
    server = StorageGRPCServer(get_minimal_modyn_config())

    request = GetRequest(dataset_id='test2', keys=['test', 'test3', 'test4'])

    for response in server.Get(request, None):
        assert response is not None
        assert response.chunk == b''


def test_get_invalid_key():
    server = StorageGRPCServer(get_minimal_modyn_config())

    request = GetRequest(dataset_id='test', keys=['test5'])

    for response in server.Get(request, None):
        assert response is not None
        assert response.chunk == b''


def test_get_not_all_keys_found():
    server = StorageGRPCServer(get_minimal_modyn_config())

    request = GetRequest(dataset_id='test', keys=['test', 'test5'])

    for response in server.Get(request, None):
        assert response is not None
        result = pickle.loads(response.chunk)
        assert isinstance(result, WebDataset)


def test_get_new_data_since():
    server = StorageGRPCServer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id='test', timestamp=0)

    response = server.GetNewDataSince(request, None)
    assert response is not None
    assert response.keys == ['test', 'test2', 'test3', 'test4']


def test_get_new_data_since_invalid_dataset():
    server = StorageGRPCServer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id='test2', timestamp=0)

    response = server.GetNewDataSince(request, None)
    assert response is not None
    assert response.keys == []


def test_get_new_data_since_now_new_data():
    server = StorageGRPCServer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id='test',
                                     timestamp=datetime.datetime.timestamp(datetime.datetime.now() +
                                                                           datetime.timedelta(seconds=1000)))

    response = server.GetNewDataSince(request, None)
    assert response is not None
    assert response.keys == []


def test_check_availability():
    server = StorageGRPCServer(get_minimal_modyn_config())

    request = DatasetAvailableRequest(dataset_id='test')

    response = server.CheckAvailability(request, None)
    assert response is not None
    assert response.available


def test_check_availability_invalid_dataset():
    server = StorageGRPCServer(get_minimal_modyn_config())

    request = DatasetAvailableRequest(dataset_id='test2')

    response = server.CheckAvailability(request, None)
    assert response is not None
    assert not response.available


def test_register_new_dataset():
    server = StorageGRPCServer(get_minimal_modyn_config())

    request = RegisterNewDatasetRequest(
        dataset_id='test3',
        base_path=os.path.dirname(TMP_FILE),
        filesystem_wrapper_type='LocalFilesystemWrapper',
        file_wrapper_type='WebdatasetFileWrapper',
        description='test',
        version='0.0.1'
    )

    response = server.RegisterNewDataset(request, None)
    assert response is not None
    assert response.success

    with DatabaseConnection(get_minimal_modyn_config()) as database:
        session = database.get_session()

        dataset = session.query(Dataset).filter(Dataset.name == 'test3').first()

        assert dataset is not None
        assert dataset.name == 'test3'
        assert dataset.base_path == os.path.dirname(TMP_FILE)
        assert dataset.description == 'test'
        assert dataset.version == '0.0.1'
