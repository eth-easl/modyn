# pylint: disable=unused-argument, no-name-in-module
import json
import os
import pathlib
from unittest.mock import patch

from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.file_wrapper.single_sample_file_wrapper import SingleSampleFileWrapper
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

TMP_FILE = str(pathlib.Path(os.path.abspath(__file__)).parent / "test.png")
TMP_FILE2 = str(pathlib.Path(os.path.abspath(__file__)).parent / "test2.png")
TMP_FILE3 = str(pathlib.Path(os.path.abspath(__file__)).parent / "test3.png")
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
                    "file_wrapper_type": SingleSampleFileWrapper,
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
    with open(TMP_FILE, "wb") as file:
        file.write(b"test")
    with open(TMP_FILE2, "wb") as file:
        file.write(b"test2")
    with open(TMP_FILE3, "wb") as file:
        file.write(b"test3")

    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        now = NOW
        before_now = now - 1

        database.create_tables()

        session = database.session

        dataset = Dataset(
            name="test",
            base_path=os.path.dirname(TMP_FILE),
            filesystem_wrapper_type="LocalFilesystemWrapper",
            file_wrapper_type="SingleSampleFileWrapper",
            description="test",
            version="0.0.1",
            file_wrapper_config=json.dumps({"file_extension": "png"}),
            last_timestamp=now,
        )

        session.add(dataset)

        session.commit()

        file = File(path=TMP_FILE, dataset=dataset, created_at=now, updated_at=now, number_of_samples=2)

        session.add(file)

        file2 = File(path=TMP_FILE2, dataset=dataset, created_at=now, updated_at=now, number_of_samples=2)

        session.add(file2)

        file3 = File(path=TMP_FILE3, dataset=dataset, created_at=before_now, updated_at=before_now, number_of_samples=2)

        session.commit()

        sample = Sample(file=file, index=0, external_key="test", label=1)

        session.add(sample)

        sample3 = Sample(file=file2, index=0, external_key="test3", label=3)

        session.add(sample3)

        sample5 = Sample(file=file3, index=0, external_key="test5", label=5)

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


def test_get():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=["test", "test3", "test5"])

    expetect_responses = [([b"test"], ["test"], [1]), ([b"test2"], ["test3"], [3])]

    for response, expetect_response in zip(server.Get(request, None), expetect_responses):
        assert response is not None
        assert response.samples == expetect_response[0]
        assert response.keys == expetect_response[1]
        assert response.labels == expetect_response[2]


def test_get_invalid_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test2", keys=["test", "test3", "test4"])

    for response in server.Get(request, None):
        assert response is not None
        assert response.samples == []
        assert response.keys == []
        assert response.labels == []


def test_get_invalid_key():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=["test5"])

    for response in server.Get(request, None):
        assert response is not None
        assert response.samples == [b"test3"]
        assert response.keys == ["test5"]
        assert response.labels == [5]


def test_get_not_all_keys_found():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=["test", "test6"])

    for response in server.Get(request, None):
        assert response is not None
        assert response.samples == [b"test"]


def test_get_no_keys_providesd():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=[])

    for response in server.Get(request, None):
        assert response is not None
        assert response.samples == []


def test_get_new_data_since():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id="test", timestamp=0)

    response = server.GetNewDataSince(request, None)
    assert response is not None
    assert response.keys == ["test", "test3", "test5"]
    assert response.timestamps == [NOW, NOW, NOW - 1]
    assert response.labels == [1, 3, 5]


def test_get_new_data_since_invalid_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id="test3", timestamp=0)

    response = server.GetNewDataSince(request, None)
    assert response is not None
    assert response.keys == []
    assert response.timestamps == []
    assert response.labels == []


def test_get_new_data_since_no_new_data():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id="test", timestamp=NOW + 100000)

    response = server.GetNewDataSince(request, None)
    assert response is not None
    assert response.keys == []
    assert response.timestamps == []
    assert response.labels == []


def test_get_data_in_interval():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetDataInIntervalRequest(dataset_id="test", start_timestamp=0, end_timestamp=NOW + 100000)

    response = server.GetDataInInterval(request, None)
    assert response is not None
    assert response.keys == ["test", "test3", "test5"]
    assert response.timestamps == [NOW, NOW, NOW - 1]
    assert response.labels == [1, 3, 5]

    request = GetDataInIntervalRequest(dataset_id="test", start_timestamp=0, end_timestamp=NOW - 1)

    response = server.GetDataInInterval(request, None)
    assert response is not None
    assert response.keys == ["test5"]
    assert response.timestamps == [NOW - 1]
    assert response.labels == [5]

    request = GetDataInIntervalRequest(dataset_id="test", start_timestamp=0, end_timestamp=10)

    response = server.GetDataInInterval(request, None)
    assert response is not None
    assert response.keys == []
    assert response.timestamps == []
    assert response.labels == []


def test_get_data_in_interval_invalid_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetDataInIntervalRequest(dataset_id="test2", start_timestamp=0, end_timestamp=NOW + 100000)

    response = server.GetDataInInterval(request, None)
    assert response is not None
    assert response.keys == []
    assert response.timestamps == []
    assert response.labels == []


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
        file_wrapper_type="SingleSampleFileWrapper",
        description="test",
        version="0.0.1",
        file_wrapper_config="{}",
    )

    response = server.RegisterNewDataset(request, None)
    assert response is not None
    assert response.success

    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        session = database.session

        dataset = session.query(Dataset).filter(Dataset.name == "test3").first()

        assert dataset is not None
        assert dataset.name == "test3"
        assert dataset.base_path == os.path.dirname(TMP_FILE)
        assert dataset.description == "test"
        assert dataset.version == "0.0.1"


@patch("modyn.storage.internal.grpc.storage_grpc_servicer.current_time_millis", return_value=NOW)
def test_get_current_timestamp(mock_current_time_millis):
    server = StorageGRPCServicer(get_minimal_modyn_config())

    response = server.GetCurrentTimestamp(None, None)
    assert response is not None
    assert response.timestamp == NOW


def test_delete_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = DatasetAvailableRequest(dataset_id="test")

    response = server.DeleteDataset(request, None)
    assert response is not None
    assert response.success

    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        session = database.session

        dataset = session.query(Dataset).filter(Dataset.name == "test").first()

        assert dataset is None
