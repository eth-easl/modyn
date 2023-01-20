import pathlib
import os
import yaml
import grpc
import time
import random 

from modyn.utils import grpc_connection_established
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.storage.internal.grpc.generated.storage_pb2 import DatasetAvailableRequest, RegisterNewDatasetRequest, GetRequest, GetNewDataSinceRequest, GetDataInIntervalRequest, GetNewDataSinceResponse
from PIL import Image

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

TIMEOUT = 120  # seconds
CONFIG_FILE = SCRIPT_PATH.parent.parent / "modyn" / "config" / "examples" / "modyn_config.yaml"
DATASET_PATH = "app" / "storage" / "datasets" / "test_dataset"

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
        print(f"Could not establish gRPC connection to storage at {storage_address}. Retrying.")
        return None

    return storage_channel


def register_new_dataset() -> None:
    storage_channel = connect_to_storage()
    if storage_channel is None:
        return
    
    storage = StorageStub(storage_channel)

    request = RegisterNewDatasetRequest(
        base_path=DATASET_PATH,
        dataset_id="test_dataset",
        description="Test dataset for integration tests.",
        file_wrapper_config="{file_extension: '.png', label_file_extension: '.txt'}",
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

    response = storage.GetCurrentTimestamp()

    assert response.timestamp > 0, "Timestamp is not valid."

def create_dataset() -> None:
    pathlib.Path(DATASET_PATH).mkdir(parents=True, exist_ok=True)


def add_image_to_dataset(image: Image, name: str) -> None:
    image.save(DATASET_PATH / name)


def create_random_image() -> Image:
    image = Image.new("RGB", (100, 100))
    random_x = random.randint(0, 100)
    random_y = random.randint(0, 100)

    random_r = random.randint(0, 255)
    random_g = random.randint(0, 255)
    random_b = random.randint(0, 255)

    image.setpixel((random_x, random_y), (random_r, random_g, random_b))

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


def get_new_data_since(timestamp: int) -> list[GetNewDataSinceResponse]:
    data = []
    storage_channel = connect_to_storage()

    if storage_channel is None:
        raise Exception("Could not connect to storage.")
    
    storage = StorageStub(storage_channel)

    request = GetNewDataSinceRequest(
        dataset_id="test_dataset",
        timestamp=timestamp,
    )

    for response in storage.GetNewDataSince(request):
        data.append(response)
    
    return data


def test_storage() -> None:
    check_get_current_timestamp() # Check if the storage service is available.
    create_dataset()
    add_images_to_dataset(0, 10) # Add images to the dataset.
    register_new_dataset()
    check_dataset_availability() # Check if the dataset is available.

    time.sleep(5) # Let the storage service process the new dataset.

    check_dataset_availability() # Check again to make sure the dataset is available.

    data = get_new_data_since(11) # Get all images currently in the dataset.

    assert len(data) == 10, "Not all images were returned."



def main() -> None:
    start_time = round(time.time())

    


if __name__ == '__main__':
    main()
