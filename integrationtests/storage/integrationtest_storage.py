import io
import json
import os
import pathlib
import random
import shutil
import time
from typing import Iterable

import grpc
import modyn.storage.internal.grpc.generated.storage_pb2 as storage_pb2
import yaml
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    GetDataInIntervalRequest,
    GetDataInIntervalResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
    GetRequest,
    RegisterNewDatasetRequest,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils import grpc_connection_established
from PIL import Image

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

TIMEOUT = 120  # seconds
CONFIG_FILE = SCRIPT_PATH.parent.parent.parent / "modyn" / "config" / "examples" / "modyn_config.yaml"
# The following path leads to a directory that is mounted into the docker container and shared with the
# storage container.
DATASET_PATH = pathlib.Path("/app") / "storage" / "datasets" / "test_dataset"

# Because we have no mapping of file to key (happens in the storage service), we have to keep
# track of the images we added to the dataset ourselves and compare them to the images we get
# from the storage service.
FIRST_ADDED_IMAGES = []
SECOND_ADDED_IMAGES = []
IMAGE_UPDATED_TIME_STAMPS = []


def get_modyn_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
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
    IMAGE_UPDATED_TIME_STAMPS.append(int(round(os.path.getmtime(DATASET_PATH / name) * 1000)))


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


def test_storage() -> None:
    check_get_current_timestamp()  # Check if the storage service is available.
    create_dataset_dir()
    add_images_to_dataset(0, 10, FIRST_ADDED_IMAGES)  # Add images to the dataset.
    register_new_dataset()
    check_dataset_availability()  # Check if the dataset is available.

    response = None
    for i in range(20):
        responses = list(get_new_data_since(0))
        assert len(responses) < 2, f"Received batched response, shouldn't happen: {responses}"
        if len(responses) == 1:
            response = responses[0]
            if len(response.keys) == 10:
                break
        time.sleep(1)

    assert response is not None, "Did not get any response from Storage"
    assert len(response.keys) == 10, f"Not all images were returned. Images returned: {response.keys}"

    check_data(response.keys, FIRST_ADDED_IMAGES)

    add_images_to_dataset(10, 20, SECOND_ADDED_IMAGES)  # Add more images to the dataset.

    for i in range(20):
        responses = list(get_new_data_since(IMAGE_UPDATED_TIME_STAMPS[9] + 1))
        assert len(responses) < 2, f"Received batched response, shouldn't happen: {responses}"
        if len(responses) == 1:
            response = responses[0]
            if len(response.keys) == 10:
                break
        time.sleep(1)

    assert response is not None, "Did not get any response from Storage"
    assert len(response.keys) == 10, f"Not all images were returned. Images returned: {response.keys}"

    check_data(response.keys, SECOND_ADDED_IMAGES)

    responses = list(get_data_in_interval(0, IMAGE_UPDATED_TIME_STAMPS[9]))
    assert len(responses) == 1, f"Received batched/no response, shouldn't happen: {responses}"
    response = responses[0]

    check_data(response.keys, FIRST_ADDED_IMAGES)

    check_get_current_timestamp()  # Check if the storage service is still available.


def main() -> None:
    try:
        test_storage()
    finally:
        cleanup_dataset_dir()
        cleanup_storage_database()


if __name__ == "__main__":
    main()
