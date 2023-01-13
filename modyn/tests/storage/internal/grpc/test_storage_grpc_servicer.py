# pylint: disable=unused-argument, no-name-in-module
import os
import pathlib
import pickle
from unittest.mock import patch

from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.file_wrapper.webdataset_file_wrapper import WebdatasetFileWrapper
from modyn.storage.internal.filesystem_wrapper.local_filesystem_wrapper import LocalFilesystemWrapper
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    GetDataInIntervalRequest,
    GetNewDataSinceRequest,
    GetRequest,
    RegisterNewDatasetRequest,
)
from modyn.storage.internal.grpc.storage_grpc_servicer import StorageGRPCServicer
from modyn.utils import current_time_millis
from webdataset import TarWriter, WebDataset

TMP_FILE = str(pathlib.Path(os.path.abspath(__file__)).parent / "test.tar")
TMP_FILE2 = str(pathlib.Path(os.path.abspath(__file__)).parent / "test2.tar")
TMP_FILE3 = str(pathlib.Path(os.path.abspath(__file__)).parent / "test3.tar")
DATABASE = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.database"
NOW = current_time_millis()


def get_minimal_modyn_config() -> dict:
    return {
        "storage": {
            "filesystem": {"type": "LocalFilesystemWrapper", "base_path": os.path.dirname(TMP_FILE)},
            "database": {
                "drivername": "sqlite",
                "username": "",
                "password": "",
                "host": "",
                "port": "0",
                "database": f"{DATABASE}",
            },
            "new_file_watcher": {"interval": 1},
            "datasets": [
                {
                    "name": "test",
                    "base_path": os.path.dirname(TMP_FILE),
                    "filesystem_wrapper_type": LocalFilesystemWrapper,
                    "file_wrapper_type": WebdatasetFileWrapper,
                    "description": "test",
                    "version": "0.0.1",
                    "file_wrapper_config": {},
                }
            ],
        },
        "project": {"name": "test", "version": "0.0.1"},
        "input": {"type": "LOCAL", "path": os.path.dirname(TMP_FILE)},
        "odm": {"type": "LOCAL"},
    }


def setup():

    os.makedirs(os.path.dirname(TMP_FILE), exist_ok=True)
    writer = TarWriter(TMP_FILE)
    writer.write({"__key__": "1", "cls": [1, 2, 3], "json": [1, 2, 3]})
    writer.write({"__key__": "2", "cls": [1, 2, 3], "json": [1, 2, 3]})
    writer.close()

    writer = TarWriter(TMP_FILE2)
    writer.write({"__key__": "3", "cls": [1, 2, 3], "json": [1, 2, 3]})
    writer.write({"__key__": "4", "cls": [1, 2, 3], "json": [1, 2, 3]})
    writer.close()

    writer = TarWriter(TMP_FILE3)
    writer.write({"__key__": "5", "cls": [1, 2, 3], "json": [1, 2, 3]})
    writer.write({"__key__": "6", "cls": [1, 2, 3], "json": [1, 2, 3]})
    writer.close()

    with DatabaseConnection(get_minimal_modyn_config()) as database:
        now = NOW
        before_now = now - 1

        database.create_all()

        session = database.get_session()

        dataset = Dataset(
            name="test",
            base_path=os.path.dirname(TMP_FILE),
            filesystem_wrapper_type="LocalFilesystemWrapper",
            file_wrapper_type="WebdatasetFileWrapper",
            description="test",
            version="0.0.1",
        )

        session.add(dataset)

        session.commit()

        file = File(path=TMP_FILE, dataset=dataset, created_at=now, updated_at=now, number_of_samples=2)

        session.add(file)

        file2 = File(path=TMP_FILE2, dataset=dataset, created_at=now, updated_at=now, number_of_samples=2)

        session.add(file2)

        file3 = File(path=TMP_FILE3, dataset=dataset, created_at=before_now, updated_at=before_now, number_of_samples=2)

        session.commit()

        sample = Sample(file=file, index=0, external_key="test")

        session.add(sample)

        sample2 = Sample(file=file, index=1, external_key="test2")

        session.add(sample2)

        sample3 = Sample(file=file2, index=0, external_key="test3")

        session.add(sample3)

        sample4 = Sample(file=file2, index=1, external_key="test4")

        session.add(sample4)

        sample5 = Sample(file=file3, index=0, external_key="test5")

        session.add(sample5)

        session.commit()


def teardown():
    os.remove(DATABASE)
    os.remove(TMP_FILE)
    os.remove(TMP_FILE2)
    os.remove(TMP_FILE3)


def test_init() -> None:
    server = StorageGRPCServicer(get_minimal_modyn_config())
    assert server is not None


@patch.object(WebdatasetFileWrapper, "get_samples_from_indices", return_value=b"")
def test_get(mock_get_samples_from_indices):
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=["test", "test3", "test4"])

    expetect_responses = [(b"", ["test"]), (b"", ["test3", "test4"])]

    for response, expetect_response in zip(server.Get(request, None), expetect_responses):
        assert response is not None
        assert response.chunk == expetect_response[0]
        assert response.keys == expetect_response[1]


def test_get_invalid_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test2", keys=["test", "test3", "test4"])

    for response in server.Get(request, None):
        assert response is not None
        assert response.chunk == b""


def test_get_invalid_key():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=["test5"])

    for response in server.Get(request, None):
        assert response is not None
        assert response.chunk == b""


def test_get_not_all_keys_found():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=["test", "test6"])

    for response in server.Get(request, None):
        assert response is not None
        result = pickle.loads(response.chunk)
        assert isinstance(result, WebDataset)


def test_get_no_keys_providesd():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=[])

    for response in server.Get(request, None):
        assert response is not None
        assert response.chunk == b""


def test_get_new_data_since():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id="test", timestamp=0)

    response = server.GetNewDataSince(request, None)
    assert response is not None
    assert response.keys == ["test", "test2", "test3", "test4", "test5"]


def test_get_new_data_since_invalid_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id="test2", timestamp=0)

    response = server.GetNewDataSince(request, None)
    assert response is not None
    assert response.keys == []


def test_get_new_data_since_no_new_data():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id="test", timestamp=NOW + 100000)

    response = server.GetNewDataSince(request, None)
    assert response is not None
    assert response.keys == []


def test_get_data_in_interval():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetDataInIntervalRequest(dataset_id="test", start_timestamp=0, end_timestamp=NOW + 100000)

    response = server.GetDataInInterval(request, None)
    assert response is not None
    assert response.keys == ["test", "test2", "test3", "test4", "test5"]

    request = GetDataInIntervalRequest(dataset_id="test", start_timestamp=0, end_timestamp=NOW - 1)

    response = server.GetDataInInterval(request, None)
    assert response is not None
    assert response.keys == ["test5"]


def test_get_data_in_interval_invalid_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetDataInIntervalRequest(dataset_id="test2", start_timestamp=0, end_timestamp=NOW + 100000)

    response = server.GetDataInInterval(request, None)
    assert response is not None
    assert response.keys == []


def test_check_availability():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = DatasetAvailableRequest(dataset_id="test")

    response = server.CheckAvailability(request, None)
    assert response is not None
    assert response.available


def test_check_availability_invalid_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = DatasetAvailableRequest(dataset_id="test2")

    response = server.CheckAvailability(request, None)
    assert response is not None
    assert not response.available


def test_register_new_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = RegisterNewDatasetRequest(
        dataset_id="test3",
        base_path=os.path.dirname(TMP_FILE),
        filesystem_wrapper_type="LocalFilesystemWrapper",
        file_wrapper_type="WebdatasetFileWrapper",
        description="test",
        version="0.0.1",
        file_wrapper_config="{}",
    )

    response = server.RegisterNewDataset(request, None)
    assert response is not None
    assert response.success

    with DatabaseConnection(get_minimal_modyn_config()) as database:
        session = database.get_session()

        dataset = session.query(Dataset).filter(Dataset.name == "test3").first()

        assert dataset is not None
        assert dataset.name == "test3"
        assert dataset.base_path == os.path.dirname(TMP_FILE)
        assert dataset.description == "test"
        assert dataset.version == "0.0.1"
