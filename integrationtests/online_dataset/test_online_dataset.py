import gc
import json
import math
import os
import pathlib
import random
import shutil
import time
from typing import Iterable, Tuple

import grpc
import modyn.storage.internal.grpc.generated.storage_pb2 as storage_pb2
import torch
import yaml
from integrationtests.utils import (
    MODYN_CONFIG_FILE,
    MODYN_DATASET_PATH,
    get_minimal_pipeline_config,
    init_metadata_db,
    register_pipeline,
)
from modyn.config.schema.pipeline.pipeline import NewDataStrategyConfig
from modyn.selector.internal.grpc.generated.selector_pb2 import DataInformRequest
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    GetDatasetSizeRequest,
    GetDatasetSizeResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
    RegisterNewDatasetRequest,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.trainer_server.internal.dataset.data_utils import prepare_dataloaders
from modyn.utils import grpc_connection_established
from modyn.utils.utils import flatten
from PIL import Image
from torchvision import transforms

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
IMAGE_UPDATED_TIME_STAMPS = []


def get_modyn_config() -> dict:
    with open(MODYN_CONFIG_FILE, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config


def connect_to_selector_servicer() -> grpc.Channel:
    selector_address = get_selector_address()
    selector_channel = grpc.insecure_channel(selector_address)

    if not grpc_connection_established(selector_channel):
        raise ConnectionError(f"Could not establish gRPC connection to selector at {selector_address}.")

    return selector_channel


def get_storage_address() -> str:
    config = get_modyn_config()
    return f"{config['storage']['hostname']}:{config['storage']['port']}"


def get_selector_address() -> str:
    config = get_modyn_config()
    return f"{config['selector']['hostname']}:{config['selector']['port']}"


def connect_to_storage() -> grpc.Channel:
    storage_address = get_storage_address()
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

    response = storage.RegisterNewDataset(request)
    assert not response.success, "Registering an existing dataset should fail"


def check_dataset_availability() -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = DatasetAvailableRequest(dataset_id="test_dataset")
    response = storage.CheckAvailability(request)

    assert response.available, "Dataset is not available."


def check_dataset_size(expected_size: int) -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)
    request = GetDatasetSizeRequest(dataset_id="test_dataset")
    response: GetDatasetSizeResponse = storage.GetDatasetSize(request)

    assert response.success, "Dataset is not available."
    assert response.num_keys == expected_size


def check_dataset_size_invalid() -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)
    request = GetDatasetSizeRequest(dataset_id="unknown_dataset")
    response: GetDatasetSizeResponse = storage.GetDatasetSize(request)

    assert not response.success, "Dataset is available (even though it should not be)."


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


def prepare_selector(num_dataworkers: int, keys: list[int]) -> Tuple[int, int]:
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)
    # We test the NewData strategy for finetuning on the new data, i.e., we reset without limit
    # We also enforce high partitioning (maximum_keys_in_memory == 2) to ensure that works

    strategy_config = NewDataStrategyConfig(
        maximum_keys_in_memory=2, limit=-1, tail_triggers=0, storage_backend="database"
    )
    pipeline_config = get_minimal_pipeline_config(max(num_dataworkers, 1), strategy_config.model_dump(by_alias=True))
    init_metadata_db(get_modyn_config())
    pipeline_id = register_pipeline(pipeline_config, get_modyn_config())

    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=keys,
            timestamps=[2 for _ in range(len(keys))],
            labels=[3 for _ in range(len(keys))],
        )
    ).trigger_id

    return pipeline_id, trigger_id


def get_new_data_since(timestamp: int) -> Iterable[GetNewDataSinceResponse]:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = GetNewDataSinceRequest(
        dataset_id="test_dataset",
        timestamp=timestamp,
    )

    responses = storage.GetNewDataSince(request)
    return responses


def get_data_keys() -> list[int]:
    keys = []
    for i in range(60):
        responses = list(get_new_data_since(0))
        keys = []
        if len(responses) > 0:
            keys = flatten([list(response.keys) for response in responses])
            if len(keys) == 10:
                break
        time.sleep(1)

    assert len(responses) > 0, "Did not get any response from Storage"
    assert len(keys) == 10, f"Not all images were returned. Images returned: {keys}"

    return keys


def get_bytes_parser() -> str:
    return """
from PIL import Image
import io
def bytes_parser_function(data: bytes) -> Image:
    return Image.open(io.BytesIO(data)).convert("RGB")"""


def tensor_in_list(tensor: torch.Tensor, tensor_list: list[torch.Tensor]) -> bool:
    return any([(tensor == c_).all() for c_ in tensor_list])


def test_dataset_impl(
    num_dataworkers: int,
    batch_size: int,
    prefetched_partitions: int,
    parallel_prefetch_requests: int,
    pipeline_id: int,
    trigger_id: int,
    items: list[int],
) -> None:
    dataloader, _ = prepare_dataloaders(
        pipeline_id,
        trigger_id,
        "test_dataset",
        num_dataworkers,
        batch_size,
        get_bytes_parser(),
        ["transforms.ToTensor()"],
        get_storage_address(),
        get_selector_address(),
        42,
        prefetched_partitions,
        parallel_prefetch_requests,
        None,
        None,
    )

    expected_min_batches = math.floor(len(items) / batch_size)
    # max one excess batch per worker
    expected_max_batches = expected_min_batches if num_dataworkers <= 1 else expected_min_batches + num_dataworkers

    all_samples = []
    all_data = []
    all_labels = []

    for batch_number, batch in enumerate(dataloader):
        sample_ids = batch[0]
        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.tolist()
        elif isinstance(sample_ids, tuple):
            sample_ids = list(sample_ids)

        assert isinstance(sample_ids, list), "Cannot parse result from DataLoader"
        assert isinstance(batch[1], torch.Tensor) and isinstance(batch[2], torch.Tensor)

        all_samples.extend(sample_ids)
        for sample in batch[1]:
            all_data.append(sample)  # iterate over batch dimension to extract samples
        all_labels.extend(batch[2].tolist())

    assert len(all_samples) == len(items)
    assert len(all_labels) == len(items)
    assert len(all_data) == len(items)

    assert expected_min_batches <= batch_number + 1 <= expected_max_batches, (
        f"[{num_dataworkers}][{batch_size}][{prefetched_partitions}]"
        + f"Wrong number of batches: {batch_number + 1}. num_items = {len(items)}."
        + f"expected_min = {expected_min_batches}, expected_max = {expected_max_batches}"
    )

    assert set(all_samples) == set(items)
    assert set(all_labels) == set(range(len(items)))

    trans = transforms.Compose([transforms.ToPILImage()])

    assert len(FIRST_ADDED_IMAGES) == len(all_data)

    for idx, image_tensor in enumerate(all_data):
        pil_image = trans(image_tensor).convert("RGB")
        image_bytes = pil_image.tobytes()
        if image_bytes not in FIRST_ADDED_IMAGES:
            raise ValueError(f"Could not find image {idx} in created images, all_samples = {all_samples}")


def test_dataset() -> None:
    NUM_IMAGES = 10

    check_get_current_timestamp()  # Check if the storage service is available.
    create_dataset_dir()
    add_images_to_dataset(0, NUM_IMAGES, FIRST_ADDED_IMAGES)  # Add images to the dataset.
    register_new_dataset()
    check_dataset_availability()  # Check if the dataset is available.
    check_dataset_size_invalid()

    keys = get_data_keys()

    for num_dataworkers in [0, 1, 2, 4, 8, 16]:
        pipeline_id, trigger_id = prepare_selector(num_dataworkers, keys)
        for prefetched_partitions in [0, 1, 2, 3, 4, 5, 999]:
            ppr_list = [999]
            if prefetched_partitions == 5:
                ppr_list = [1, 2, 5, 999]

            for parallel_prefetch_requests in ppr_list:
                for batch_size in [1, 2, 10]:
                    print(
                        f"Testing num_workers = {num_dataworkers}, partitions = {prefetched_partitions},"
                        + f"batch_size = {batch_size}, parallel_prefetch_requests={parallel_prefetch_requests}"
                    )
                    test_dataset_impl(
                        num_dataworkers,
                        batch_size,
                        prefetched_partitions,
                        parallel_prefetch_requests,
                        pipeline_id,
                        trigger_id,
                        keys,
                    )
                    gc.collect()


def main() -> None:
    try:
        test_dataset()
    finally:
        cleanup_storage_database()
        cleanup_dataset_dir()


if __name__ == "__main__":
    main()
