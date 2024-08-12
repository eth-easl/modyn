import io
import json
import math
import os
import pathlib
import random
import shutil
import time
from collections.abc import Iterable

import grpc
import yaml
from PIL import Image

import modyn.storage.internal.grpc.generated.storage_pb2 as storage_pb2
from integrationtests.utils import MODYN_CONFIG_FILE, MODYN_DATASET_PATH
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    DeleteDataRequest,
    GetDataInIntervalRequest,
    GetDataInIntervalResponse,
    GetDataPerWorkerRequest,
    GetDataPerWorkerResponse,
    GetDatasetSizeRequest,
    GetDatasetSizeResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
    GetRequest,
    RegisterNewDatasetRequest,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils import grpc_connection_established
from modyn.utils.utils import flatten

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

TIMEOUT = 120  # seconds

# The following path leads to a directory that is mounted into the docker container and shared with the
# storage container.
DATASET_PATH = MODYN_DATASET_PATH / "test_dataset"

# Because we have no mapping of file to key (happens in the storage service), we have to keep
# track of the images we added to the dataset ourselves and compare them to the images we get
# from the storage service.
FIRST_ADDED_IMAGES = []
SECOND_ADDED_IMAGES = []
THIRD_ADDED_IMAGES = []
IMAGE_UPDATED_TIME_STAMPS = []


def get_modyn_config() -> dict:
    with open(MODYN_CONFIG_FILE, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config


def connect_to_storage() -> grpc.Channel:
    config = get_modyn_config()

    storage_address = f"{config['storage']['hostname']}:{config['storage']['port']}"
    storage_channel = grpc.insecure_channel(storage_address)

    if not grpc_connection_established(storage_channel) or storage_channel is None:
        raise ConnectionError(f"Could not establish gRPC connection to storage at {storage_address}.")

    return storage_channel


def register_new_dataset() -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = RegisterNewDatasetRequest(
        base_path=str(DATASET_PATH),
        dataset_id="test_dataset",
        description="Test dataset for integration tests.",
        file_wrapper_config=json.dumps({"file_extension": ".png", "label_file_extension": ".txt"}),
        file_wrapper_type="SingleSampleFileWrapper",
        filesystem_wrapper_type="LocalFilesystemWrapper",
        version="0.1.0",
    )

    response = storage.RegisterNewDataset(request)
    assert response.success, "Could not register new dataset."


def check_dataset_availability() -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = DatasetAvailableRequest(dataset_id="test_dataset")
    response = storage.CheckAvailability(request)

    assert response.available, "Dataset is not available."


def check_dataset_size(expected_size: int, start_timestamp=None, end_timestamp=None) -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)
    request = GetDatasetSizeRequest(
        dataset_id="test_dataset", start_timestamp=start_timestamp, end_timestamp=end_timestamp
    )
    response: GetDatasetSizeResponse = storage.GetDatasetSize(request)

    assert response.success, "Dataset is not available."
    assert response.num_keys == expected_size


def check_dataset_size_invalid() -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)
    request = GetDatasetSizeRequest(dataset_id="unknown_dataset")
    response: GetDatasetSizeResponse = storage.GetDatasetSize(request)

    assert not response.success, "Dataset is available (even though it should not be)."


def check_data_per_worker() -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    # 30 images in total; the first two workers should get 5 images each, the rest should get 4 images
    for worker_id in range(7):
        request = GetDataPerWorkerRequest(dataset_id="test_dataset", worker_id=worker_id, total_workers=7)
        responses: list[GetDataPerWorkerResponse] = list(storage.GetDataPerWorker(request))

        assert len(responses) == 1, f"Received batched response or no response, shouldn't happen: {responses}"

        response_keys_size = len(responses[0].keys)

        assert response_keys_size == 5 if worker_id <= 1 else response_keys_size == 4

    split_ts1 = IMAGE_UPDATED_TIME_STAMPS[9] + 1
    split_ts2 = IMAGE_UPDATED_TIME_STAMPS[19] + 1

    for worker_id in range(3):
        request = GetDataPerWorkerRequest(
            dataset_id="test_dataset", worker_id=worker_id, total_workers=3, end_timestamp=split_ts2
        )
        responses: list[GetDataPerWorkerResponse] = list(storage.GetDataPerWorker(request))

        alternative_request = GetDataPerWorkerRequest(
            dataset_id="test_dataset", worker_id=worker_id, total_workers=3, start_timestamp=0, end_timestamp=split_ts2
        )
        alternative_responses: list[GetDataPerWorkerResponse] = list(storage.GetDataPerWorker(alternative_request))

        assert alternative_responses == responses

        assert len(responses) == 1, f"Received batched response or no response, shouldn't happen: {responses}"

        response_keys_size = len(responses[0].keys)

        assert response_keys_size == 7 if worker_id <= 1 else response_keys_size == 6

    for worker_id in range(3):
        request = GetDataPerWorkerRequest(
            dataset_id="test_dataset", worker_id=worker_id, total_workers=3, start_timestamp=split_ts2
        )
        responses: list[GetDataPerWorkerResponse] = list(storage.GetDataPerWorker(request))

        assert len(responses) == 1, f"Received batched response or no response, shouldn't happen: {responses}"

        response_keys_size = len(responses[0].keys)

        assert response_keys_size == 4 if worker_id == 0 else response_keys_size == 3

    for worker_id in range(3):
        request = GetDataPerWorkerRequest(
            dataset_id="test_dataset",
            worker_id=worker_id,
            total_workers=3,
            start_timestamp=split_ts1,
            end_timestamp=split_ts2,
        )
        responses: list[GetDataPerWorkerResponse] = list(storage.GetDataPerWorker(request))

        assert len(responses) == 1, f"Received batched response or no response, shouldn't happen: {responses}"

        response_keys_size = len(responses[0].keys)

        assert response_keys_size == 4 if worker_id == 0 else response_keys_size == 3


def check_get_current_timestamp() -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    empty = storage_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    response = storage.GetCurrentTimestamp(empty)

    assert response.timestamp > 0, "Timestamp is not valid."


def create_dataset_dir() -> None:
    pathlib.Path(DATASET_PATH).mkdir(parents=True, exist_ok=True)


def cleanup_dataset_dir() -> None:
    shutil.rmtree(DATASET_PATH)


def cleanup_storage_database() -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = DatasetAvailableRequest(dataset_id="test_dataset")

    response = storage.DeleteDataset(request)

    assert response.success, "Could not cleanup storage database."


def add_image_to_dataset(image: Image, name: str) -> None:
    image.save(DATASET_PATH / name)
    IMAGE_UPDATED_TIME_STAMPS.append(int(math.floor(os.path.getmtime(DATASET_PATH / name))))


def create_random_image() -> Image:
    image = Image.new("RGB", (100, 100))
    random_x = random.randint(0, 99)
    random_y = random.randint(0, 99)

    random_r = random.randint(0, 254)
    random_g = random.randint(0, 254)
    random_b = random.randint(0, 254)

    image.putpixel((random_x, random_y), (random_r, random_g, random_b))

    return image


def add_images_to_dataset(start_number: int, end_number: int, images_added: list[bytes]) -> None:
    create_dataset_dir()

    for i in range(start_number, end_number):
        image = create_random_image()
        add_image_to_dataset(image, f"image_{i}.png")
        images_added.append(image.tobytes())
        with open(DATASET_PATH / f"image_{i}.txt", "w") as label_file:
            label_file.write(f"{i}")


def get_new_data_since(timestamp: int) -> Iterable[GetNewDataSinceResponse]:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = GetNewDataSinceRequest(
        dataset_id="test_dataset",
        timestamp=timestamp,
    )

    responses = storage.GetNewDataSince(request)

    return responses


def get_data_in_interval(start_timestamp: int, end_timestamp: int) -> Iterable[GetDataInIntervalResponse]:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = GetDataInIntervalRequest(
        dataset_id="test_dataset",
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )

    responses = storage.GetDataInInterval(request)

    return responses


def check_data(keys: list[str], expected_images: list[bytes]) -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = GetRequest(
        dataset_id="test_dataset",
        keys=keys,
    )

    i = -1
    for i, response in enumerate(storage.Get(request)):
        if len(response.samples) == 0:
            assert False, f"Could not get image with key {keys[i]}."
        for sample in response.samples:
            if sample is None:
                assert False, f"Could not get image with key {keys[i]}."
            image = Image.open(io.BytesIO(sample))
            if image.tobytes() not in expected_images:
                raise ValueError(f"Image with key {keys[i]} is not present in the expected images.")
    assert i == len(keys) - 1, f"Could not get all images. Images missing: keys: {keys} i: {i}"


def check_delete_data(keys_to_delete: list[int]) -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = DeleteDataRequest(
        dataset_id="test_dataset",
        keys=keys_to_delete,
    )

    responses = storage.DeleteData(request)

    assert responses.success, "Could not delete data."


def test_storage() -> None:
    check_get_current_timestamp()  # Check if the storage service is available.
    create_dataset_dir()
    register_new_dataset()
    check_dataset_availability()  # Check if the dataset is available.
    check_dataset_size(0)  # Check if the dataset is empty.

    check_dataset_size_invalid()

    add_images_to_dataset(0, 10, FIRST_ADDED_IMAGES)  # Add images to the dataset.

    for i in range(20):
        keys = []
        labels = []
        responses = list(get_new_data_since(0))
        if len(responses) > 0:
            keys = flatten([list(response.keys) for response in responses])
            labels = flatten([list(response.keys) for response in responses])
            if len(keys) == 10:
                assert (label in [f"{i}" for i in range(0, 10)] for label in labels)
                break
        time.sleep(1)

    assert len(responses) > 0, "Did not get any response from Storage"
    assert len(keys) == 10, "Not all images were returned."

    first_image_keys = keys

    check_data(keys, FIRST_ADDED_IMAGES)
    check_dataset_size(10)

    # Otherwise, if the test runs too quick, the timestamps of the new data equals the timestamps of the old data, and then we have a problem
    print("Sleeping for 2 seconds before adding more images to the dataset...")
    time.sleep(2)
    print("Continuing test.")

    add_images_to_dataset(10, 20, SECOND_ADDED_IMAGES)  # Add more images to the dataset.

    for i in range(60):
        keys = []
        labels = []
        responses = list(get_new_data_since(IMAGE_UPDATED_TIME_STAMPS[9] + 1))
        if len(responses) > 0:
            keys = flatten([list(response.keys) for response in responses])
            labels = flatten([list(response.keys) for response in responses])
            if len(keys) == 10:
                assert (label in [f"{i}" for i in range(10, 20)] for label in labels)
                break
        time.sleep(1)

    assert len(responses) > 0, "Did not get any response from Storage"
    assert len(keys) == 10, f"Not all images were returned. Images returned = {keys}"

    check_data(keys, SECOND_ADDED_IMAGES)
    check_dataset_size(20)

    responses = list(get_data_in_interval(0, IMAGE_UPDATED_TIME_STAMPS[9]))

    assert len(responses) > 0, f"Received no response, shouldn't happen: {responses}"
    keys = flatten([list(response.keys) for response in responses])

    check_data(keys, FIRST_ADDED_IMAGES)

    print("Sleeping for 2 seconds before adding even more images to the dataset...")
    time.sleep(2)
    print("Continuing test.")
    add_images_to_dataset(20, 30, THIRD_ADDED_IMAGES)

    for i in range(60):
        keys = []
        labels = []
        responses = list(get_new_data_since(IMAGE_UPDATED_TIME_STAMPS[19] + 1))
        if len(responses) > 0:
            keys = flatten([list(response.keys) for response in responses])
            labels = flatten([list(response.keys) for response in responses])
            if len(keys) == 10:
                assert (label in [f"{i}" for i in range(20, 30)] for label in labels)
                break
        time.sleep(1)

    assert len(responses) > 0, "Did not get any response from Storage"
    assert len(keys) == 10, f"Not all images were returned. Images returned = {keys}"

    check_data_per_worker()
    check_dataset_size(30)
    check_dataset_size(10, start_timestamp=IMAGE_UPDATED_TIME_STAMPS[19] + 1)
    # a sanity check for 0 as end_timestamp
    check_dataset_size(0, start_timestamp=IMAGE_UPDATED_TIME_STAMPS[19] + 1, end_timestamp=0)
    # this check can be seen as duplicate as the previous one,
    # but we want to ensure setting start_timestamp as 0 or None has the same effect
    check_dataset_size(20, start_timestamp=0, end_timestamp=IMAGE_UPDATED_TIME_STAMPS[19] + 1)
    check_dataset_size(20, end_timestamp=IMAGE_UPDATED_TIME_STAMPS[19] + 1)
    check_dataset_size(
        10, start_timestamp=IMAGE_UPDATED_TIME_STAMPS[9] + 1, end_timestamp=IMAGE_UPDATED_TIME_STAMPS[19] + 1
    )

    check_delete_data(first_image_keys)

    check_dataset_size(20)

    check_get_current_timestamp()  # Check if the storage service is still available.


def main() -> None:
    try:
        test_storage()
    finally:
        cleanup_storage_database()
        cleanup_dataset_dir()


if __name__ == "__main__":
    main()
