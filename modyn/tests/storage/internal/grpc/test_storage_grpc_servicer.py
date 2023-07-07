# pylint: disable=unused-argument, no-name-in-module
import json
import os
import pathlib
from unittest.mock import patch

import pytest
from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.file_wrapper.single_sample_file_wrapper import SingleSampleFileWrapper
from modyn.storage.internal.filesystem_wrapper.local_filesystem_wrapper import LocalFilesystemWrapper
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    DeleteDataRequest,
    GetDataInIntervalRequest,
    GetDataPerWorkerRequest,
    GetDataPerWorkerResponse,
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
            "sample_batch_size": 1024,
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
    if os.path.exists(DATABASE):
        os.remove(DATABASE)

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

        session.add(file3)

        session.commit()

        sample = Sample(dataset_id=dataset.dataset_id, file_id=file.file_id, index=0, label=1)

        session.add(sample)

        sample3 = Sample(dataset_id=dataset.dataset_id, file_id=file2.file_id, index=0, label=3)

        session.add(sample3)

        sample5 = Sample(dataset_id=dataset.dataset_id, file_id=file3.file_id, index=0, label=5)

        session.add(sample5)

        session.commit()

        assert (
            sample.sample_id == 1 and sample3.sample_id == 2 and sample5.sample_id == 3
        ), "Inherent assumptions of primary key generation not met"


def teardown():
    os.remove(DATABASE)
    try:
        os.remove(TMP_FILE)
    except FileNotFoundError:
        pass
    try:
        os.remove(TMP_FILE2)
    except FileNotFoundError:
        pass
    try:
        os.remove(TMP_FILE3)
    except FileNotFoundError:
        pass


def test_init() -> None:
    server = StorageGRPCServicer(get_minimal_modyn_config())
    assert server is not None


def test_get():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=[0, 1, 2])

    expected_responses = [([b"test"], [1], [1]), ([b"test2"], [2], [3]), ([b"test3"], [3], [5])]

    for response, expected_response in zip(server.Get(request, None), expected_responses):
        assert response is not None
        assert response.samples == expected_response[0]
        assert response.keys == expected_response[1]
        assert response.labels == expected_response[2]


def test_get_invalid_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test2", keys=[1, 2, 3])

    for response in server.Get(request, None):
        assert response is not None
        assert response.samples == []
        assert response.keys == []
        assert response.labels == []


def test_get_invalid_key():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=[42])
    responses = list(server.Get(request, None))
    assert len(responses) == 1
    response = responses[0]

    assert response is not None
    assert response.samples == []
    assert response.keys == []
    assert response.labels == []


def test_get_not_all_keys_found():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetRequest(dataset_id="test", keys=[1, 42])

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

    responses = list(server.GetNewDataSince(request, None))
    assert 1 == len(responses)
    response = responses[0]

    assert response is not None
    assert response.keys == [3, 1, 2]
    assert response.timestamps == [NOW - 1, NOW, NOW]
    assert response.labels == [5, 1, 3]


def test_get_new_data_since_batched():
    server = StorageGRPCServicer(get_minimal_modyn_config())
    server._sample_batch_size = 1

    request = GetNewDataSinceRequest(dataset_id="test", timestamp=0)

    responses = list(server.GetNewDataSince(request, None))

    assert 3 == len(responses)
    response1 = responses[0]
    response2 = responses[1]
    response3 = responses[2]

    assert response1 is not None
    assert response1.keys == [3]
    assert response1.timestamps == [NOW - 1]
    assert response1.labels == [5]

    assert response2 is not None
    assert response2.keys == [1]
    assert response2.timestamps == [NOW]
    assert response2.labels == [1]

    assert response3 is not None
    assert response3.keys == [2]
    assert response3.timestamps == [NOW]
    assert response3.labels == [3]


def test_get_new_data_since_invalid_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id="test3", timestamp=0)

    responses = list(server.GetNewDataSince(request, None))
    assert len(responses) == 1
    response = responses[0]
    assert response is not None
    assert response.keys == []
    assert response.timestamps == []
    assert response.labels == []


def test_get_new_data_since_no_new_data():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetNewDataSinceRequest(dataset_id="test", timestamp=NOW + 100000)

    responses = list(server.GetNewDataSince(request, None))
    assert len(responses) == 0


def test_get_data_in_interval():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetDataInIntervalRequest(dataset_id="test", start_timestamp=0, end_timestamp=NOW + 100000)

    responses = list(server.GetDataInInterval(request, None))

    assert len(responses) == 1
    response = responses[0]

    assert response is not None
    assert response.keys == [3, 1, 2]
    assert response.timestamps == [NOW - 1, NOW, NOW]
    assert response.labels == [5, 1, 3]

    request = GetDataInIntervalRequest(dataset_id="test", start_timestamp=0, end_timestamp=NOW - 1)

    responses = list(server.GetDataInInterval(request, None))

    assert len(responses) == 1
    response = responses[0]

    assert response is not None
    assert response.keys == [3]
    assert response.timestamps == [NOW - 1]
    assert response.labels == [5]

    request = GetDataInIntervalRequest(dataset_id="test", start_timestamp=0, end_timestamp=10)

    responses = list(server.GetDataInInterval(request, None))

    assert len(responses) == 0


def test_get_data_in_interval_invalid_dataset():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetDataInIntervalRequest(dataset_id="test2", start_timestamp=0, end_timestamp=NOW + 100000)

    responses = list(server.GetDataInInterval(request, None))
    assert len(responses) == 1
    response = responses[0]
    assert response is not None
    assert response.keys == []
    assert response.timestamps == []
    assert response.labels == []


def test_get_data_per_worker():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = GetDataPerWorkerRequest(dataset_id="test", worker_id=0, total_workers=2)
    response: [GetDataPerWorkerResponse] = list(server.GetDataPerWorker(request, None))
    assert len(response) == 1
    assert response[0].keys == [1, 2]

    request = GetDataPerWorkerRequest(dataset_id="test", worker_id=1, total_workers=2)
    response = list(server.GetDataPerWorker(request, None))
    assert len(response) == 1
    assert response[0].keys == [3]

    request = GetDataPerWorkerRequest(dataset_id="test", worker_id=3, total_workers=4)
    response: [GetDataPerWorkerResponse] = list(server.GetDataPerWorker(request, None))
    assert len(response) == 0

    request = GetDataPerWorkerRequest(dataset_id="test", worker_id=0, total_workers=1)
    response: [GetDataPerWorkerResponse] = list(server.GetDataPerWorker(request, None))
    assert len(response) == 1
    assert response[0].keys == [1, 2, 3]

    request = GetDataPerWorkerRequest(dataset_id="test", worker_id=2, total_workers=2)
    with pytest.raises(ValueError):
        list(server.GetDataPerWorker(request, None))


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


def test_delete_data():
    server = StorageGRPCServicer(get_minimal_modyn_config())

    request = DeleteDataRequest(dataset_id="test", keys=[1, 2])

    response = server.DeleteData(request, None)
    assert response is not None
    assert response.success

    assert not os.path.exists(TMP_FILE)
    assert not os.path.exists(TMP_FILE2)
    assert os.path.exists(TMP_FILE3)

    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        session = database.session

        files = session.query(File).filter(File.dataset_id == "test").all()

        assert len(files) == 0


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
