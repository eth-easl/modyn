import pathlib
import os
import yaml
import grpc
import time
import random 
import json
import io

from modyn.utils import grpc_connection_established
import modyn.storage.internal.grpc.generated.storage_pb2 as storage_pb2
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.storage.internal.grpc.generated.storage_pb2 import DatasetAvailableRequest, RegisterNewDatasetRequest, GetRequest, GetNewDataSinceRequest, GetDataInIntervalRequest, GetNewDataSinceResponse, GetDataInIntervalResponse
from PIL import Image

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

TIMEOUT = 120  # seconds
CONFIG_FILE = SCRIPT_PATH.parent.parent.parent / "modyn" / "config" / "examples" / "modyn_config.yaml"
DATASET_PATH =  pathlib.Path("/app") / "storage" / "datasets" / "test_dataset"

IMAGES = []


def get_modyn_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config


def connect_to_storage() -> grpc.Channel:
    config = get_modyn_config()

    storage_address = f"{config['storage']['hostname']}:{config['storage']['port']}"
    storage_channel = grpc.insecure_channel(storage_address)

    if not grpc_connection_established(storage_channel):
        assert False, (f"Could not establish gRPC connection to storage at {storage_address}.")

    return storage_channel


def register_new_dataset() -> None:
    storage_channel = connect_to_storage()
    if storage_channel is None:
        return
    
    storage = StorageStub(storage_channel)

    request = RegisterNewDatasetRequest(
        base_path=str(DATASET_PATH),
        dataset_id="test_dataset",
        description="Test dataset for integration tests.",
        file_wrapper_config=json.dumps({"file_extension": '.png', "label_file_extension": '.txt'}),
        file_wrapper_type="SingleSampleFileWrapper",
        filesystem_wrapper_type="LocalFilesystemWrapper",
        version="0.1.0",
    )

    response = storage.RegisterNewDataset(request)

    assert response.success, "Could not register new dataset."


def check_dataset_availability() -> None:
    storage_channel = connect_to_storage()
    if storage_channel is None:
        return False

    storage = StorageStub(storage_channel)

    request = DatasetAvailableRequest(dataset_id="test_dataset")
    response = storage.CheckAvailability(request)

    assert response.available, "Dataset is not available."


def check_get_current_timestamp() -> None:
    storage_channel = connect_to_storage()
    if storage_channel is None:
        return False

    storage = StorageStub(storage_channel)

    empty = storage_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    response = storage.GetCurrentTimestamp(empty)

    assert response.timestamp > 0, "Timestamp is not valid."

def create_dataset() -> None:
    pathlib.Path(DATASET_PATH).mkdir(parents=True, exist_ok=True)


def add_image_to_dataset(image: Image, name: str) -> None:
    image.save(DATASET_PATH / name)


def create_random_image() -> Image:
    image = Image.new("RGB", (100, 100))
    random_x = random.randint(0, 99)
    random_y = random.randint(0, 99)

    random_r = random.randint(0, 254)
    random_g = random.randint(0, 254)
    random_b = random.randint(0, 254)

    image.putpixel((random_x, random_y), (random_r, random_g, random_b))

    return image


def set_timestamp(file: str, timestamp: int) -> None:
    os.utime(file, (timestamp, timestamp))


def add_images_to_dataset(start_number: int, end_number: int) -> None:
    create_dataset()

    for i in range(start_number, end_number):
        image = create_random_image()
        add_image_to_dataset(image, f"image_{i}.png")
        IMAGES.append(image)
        set_timestamp(DATASET_PATH / f"image_{i}.png", i)
        with open(DATASET_PATH / f"image_{i}.txt", "w") as label_file:
            label_file.write(f"{i}")


def get_new_data_since(timestamp: int) -> GetNewDataSinceResponse:
    storage_channel = connect_to_storage()

    if storage_channel is None:
        raise Exception("Could not connect to storage.")
    
    storage = StorageStub(storage_channel)

    request = GetNewDataSinceRequest(
        dataset_id="test_dataset",
        timestamp=timestamp,
    )

    response = storage.GetNewDataSince(request)
    
    return response


def get_data_in_interval(start_timestamp: int, end_timestamp: int) -> GetDataInIntervalResponse:
    storage_channel = connect_to_storage()

    if storage_channel is None:
        raise Exception("Could not connect to storage.")
    
    storage = StorageStub(storage_channel)

    request = GetDataInIntervalRequest(
        dataset_id="test_dataset",
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )

    response = storage.GetDataInInterval(request)
    
    return response


def check_data(keys: list[str], indices: list[int]) -> None:

    storage_channel = connect_to_storage()

    if storage_channel is None:
        raise Exception("Could not connect to storage.")
    
    storage = StorageStub(storage_channel)

    request = GetRequest(
            dataset_id="test_dataset",
            keys=keys,
    )

    expected_images = [IMAGES[index].tobytes() for index in indices]

    for i, response in enumerate(storage.Get(request)):
        if response.chunk is None:
            assert False, f"Could not get image with key {keys[i]}."
        image = Image.open(io.BytesIO(response.chunk))
        if image.tobytes() not in expected_images:
            assert False, f"Image with key {keys[i]} is not equal to the image in the dataset. Image returned: {image.tobytes()}"
        expected_images.remove(image.tobytes())
    if len(expected_images) > 0:
        assert False, f"Could not get all images. Images missing: {expected_images}"


def test_storage() -> None:
    check_get_current_timestamp() # Check if the storage service is available.
    create_dataset()
    add_images_to_dataset(0, 10) # Add images to the dataset.
    register_new_dataset()
    check_dataset_availability() # Check if the dataset is available.

    time.sleep(20) # Let the storage service process the new dataset.

    check_dataset_availability() # Check again to make sure the dataset is available.

    response = get_new_data_since(0) # Get all images currently in the dataset.

    assert len(response.keys) == 10, f"Not all images were returned. Images returned: {response.keys}"
    check_data(response.keys, list(range(0, 10)))

    add_images_to_dataset(10, 20) # Add more images to the dataset.

    time.sleep(20) # Let the storage service process the new images.

    response = get_new_data_since(10000) # Get all images currently in the dataset.

    assert len(response.keys) == 20, f"Not all images were returned. Images returned: {response.keys}"
    check_data(response.keys, list(range(10, 20)))

    response = get_data_in_interval(5000, 15000) # Get all images currently in the dataset.

    assert len(response.keys) == 11, f"Not all images were returned. Images returned: {response.keys}"
    check_data(response.keys, list(range(5, 16)))

    check_get_current_timestamp() # Check if the storage service is still available.


def main() -> None:
    test_storage()

if __name__ == '__main__':
    main()
