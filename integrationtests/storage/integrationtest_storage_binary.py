############
# storage integration tests adapted to binary input format.
# Unchanged functions are imported from the original test
# Instead of images, we have binary files. The binary files with random content of size 10 bytes.

import json
import math
import os
import random
import time

# unchanged functions are imported from the original test file
from integrationtests.storage.integrationtest_storage import (
    DATASET_PATH,
    check_dataset_availability,
    check_get_current_timestamp,
    cleanup_dataset_dir,
    cleanup_storage_database,
    connect_to_storage,
    create_dataset_dir,
    get_data_in_interval,
    get_new_data_since,
)
from modyn.storage.internal.grpc.generated.storage_pb2 import GetRequest, RegisterNewDatasetRequest
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils.utils import flatten

# Because we have no mapping of file to key (happens in the storage service), we have to keep
# track of the samples we added to the dataset ourselves and compare them to the samples we get
# from the storage service.
FIRST_ADDED_BINARY = []
SECOND_ADDED_BINARY = []
BINARY_UPDATED_TIME_STAMPS = []


def register_new_dataset() -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = RegisterNewDatasetRequest(
        base_path=str(DATASET_PATH),
        dataset_id="test_dataset",
        description="Test dataset for integration tests of binary wrapper.",
        file_wrapper_config=json.dumps(
            {
                "byteorder": "little",
                "file_extension": ".bin",
                "label_size": 4,
                "record_size": 10,
            }
        ),
        file_wrapper_type="BinaryFileWrapper",
        filesystem_wrapper_type="LocalFilesystemWrapper",
        version="0.1.0",
    )

    response = storage.RegisterNewDataset(request)

    assert response.success, "Could not register new dataset."


def add_file_to_dataset(binary_data: bytes, name: str) -> None:
    with open(DATASET_PATH / name, "wb") as f:
        f.write(binary_data)
    BINARY_UPDATED_TIME_STAMPS.append(int(math.floor(os.path.getmtime(DATASET_PATH / name))))


def create_random_binary_file() -> tuple[bytes, list[bytes]]:
    binary_data = b""
    samples = []
    for i in range(250):
        sample_binary_data = random.randbytes(10)
        binary_data += sample_binary_data
        samples.append(sample_binary_data[4:])

    return binary_data, samples


def add_files_to_dataset(
    start_number: int,
    end_number: int,
    samples: list[bytes],
) -> list[bytes]:
    create_dataset_dir()

    for i in range(start_number, end_number):
        binary_file, file_samples = create_random_binary_file()
        add_file_to_dataset(binary_file, f"binary_{i}.bin")
        samples.extend(file_samples)

    return samples


def check_data(keys: list[str], expected_samples: list[bytes]) -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = GetRequest(
        dataset_id="test_dataset",
        keys=keys,
    )
    samples_counter = 0
    for _, response in enumerate(storage.Get(request)):
        if len(response.samples) == 0:
            assert False, f"Could not get sample with key {keys[samples_counter]}."
        for sample in response.samples:
            if sample is None:
                assert False, f"Could not get sample with key {keys[samples_counter]}."
            if sample not in expected_samples:
                raise ValueError(
                    f"Sample {sample} with key {keys[samples_counter]} is not present in the "
                    f"expected samples {expected_samples}. "
                )
            samples_counter += 1

    assert samples_counter == len(
        keys
    ), f"Could not get all samples. Samples missing: keys: {sorted(keys)} i: {samples_counter}"


def test_storage() -> None:
    check_get_current_timestamp()  # Check if the storage service is available.
    create_dataset_dir()
    register_new_dataset()
    check_dataset_availability()  # Check if the dataset is available.

    add_files_to_dataset(0, 10, FIRST_ADDED_BINARY)  # Add samples to the dataset.

    responses = None
    for i in range(500):
        responses = list(get_new_data_since(0))
        keys = []
        if len(responses) > 0:
            keys = flatten([list(response.keys) for response in responses])
            if len(keys) == 2500:  # 10 files, each one with 250 samples
                break
        time.sleep(1)

    assert len(responses) > 0, "Did not get any response from Storage"
    assert len(keys) == 2500, f"Not all samples were returned. Samples returned: {keys}"

    check_data(keys, FIRST_ADDED_BINARY)

    # Otherwise, if the test runs too quick, the timestamps of the new data equals the timestamps of the old data, and then we have a problem
    print("Sleeping for 2 seconds before adding more binary files to the dataset...")
    time.sleep(2)
    print("Continuing test.")

    add_files_to_dataset(10, 20, SECOND_ADDED_BINARY)  # Add more samples to the dataset.

    for i in range(500):
        responses = list(get_new_data_since(BINARY_UPDATED_TIME_STAMPS[9] + 1))
        keys = []
        if len(responses) > 0:
            keys = flatten([list(response.keys) for response in responses])
            if len(keys) == 2500:
                break
        time.sleep(1)

    assert len(responses) > 0, "Did not get any response from Storage"
    assert len(keys) == 2500, f"Not all samples were returned. Samples returned: {keys}"

    check_data(keys, SECOND_ADDED_BINARY)

    responses = list(get_data_in_interval(0, BINARY_UPDATED_TIME_STAMPS[9]))
    assert len(responses) > 0, f"Received no response, shouldn't happen: {responses}"
    keys = flatten([list(response.keys) for response in responses])

    check_data(keys, FIRST_ADDED_BINARY)

    check_get_current_timestamp()  # Check if the storage service is still available.


def main() -> None:
    try:
        test_storage()
    finally:
        cleanup_storage_database()
        cleanup_dataset_dir()


if __name__ == "__main__":
    main()
