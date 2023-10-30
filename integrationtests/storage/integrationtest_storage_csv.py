############
# storage integration tests adapted to CSV input format.
# Unchanged functions are imported from the original test
# Instead of images, we have CSV files. Each file has 25 rows end each row has 5 columns.
# f"A{index}file{file},B{index}file{file},C{index}file{file},{counter}"
# where index is a random number, file is the fileindex and the label (last column) is a global counter

import json
import os
import random
import time
from typing import Tuple

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

# Because we have no mapping of file to key (happens in the storage service), we have to keep
# track of the samples we added to the dataset ourselves and compare them to the samples we get
# from the storage service.
FIRST_ADDED_CSVS = []
SECOND_ADDED_CSVS = []
CSV_UPDATED_TIME_STAMPS = []


def register_new_dataset() -> None:
    storage_channel = connect_to_storage()

    storage = StorageStub(storage_channel)

    request = RegisterNewDatasetRequest(
        base_path=str(DATASET_PATH),
        dataset_id="test_dataset",
        description="Test dataset for integration tests of CSV wrapper.",
        file_wrapper_config=json.dumps({"file_extension": ".csv", "separator": ",", "label_index": 3}),
        file_wrapper_type="CsvFileWrapper",
        filesystem_wrapper_type="LocalFilesystemWrapper",
        version="0.1.0",
    )

    response = storage.RegisterNewDataset(request)

    assert response.success, "Could not register new dataset."


def add_file_to_dataset(csv_file_content: str, name: str) -> None:
    with open(DATASET_PATH / name, "w") as f:
        f.write(csv_file_content)
    CSV_UPDATED_TIME_STAMPS.append(int(round(os.path.getmtime(DATASET_PATH / name) * 1000)))


def create_random_csv_row(file: int, counter: int) -> str:
    index = random.randint(1, 1000)
    return f"A{index}file{file},B{index}file{file},C{index}file{file},{counter}"


def create_random_csv_file(file: int, counter: int) -> Tuple[str, list[str], int]:
    rows = []
    samples = []
    for repeat in range(25):
        row = create_random_csv_row(file, counter)
        counter += 1
        rows.append(row)
        sample = ",".join(row.split(",")[:3])  # remove the label
        samples.append(sample)

    return "\n".join(rows), samples, counter


def add_files_to_dataset(start_number: int, end_number: int, files_added: list[bytes], rows_added: list[bytes]) -> None:
    create_dataset_dir()
    counter = 0
    for i in range(start_number, end_number):
        csv_file, samples_csv_file, counter = create_random_csv_file(i, counter)
        add_file_to_dataset(csv_file, f"csv_{i}.csv")
        files_added.append(bytes(csv_file, "utf-8"))
        [rows_added.append(bytes(row, "utf-8")) for row in samples_csv_file]


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

    add_files_to_dataset(0, 10, [], FIRST_ADDED_CSVS)  # Add samples to the dataset.

    response = None
    for i in range(500):
        responses = list(get_new_data_since(0))
        assert len(responses) < 2, f"Received batched response, shouldn't happen: {responses}"
        if len(responses) == 1:
            response = responses[0]
            if len(response.keys) == 250:  # 10 files, each one with 250 samples
                break
        time.sleep(1)

    assert response is not None, "Did not get any response from Storage"
    assert len(response.keys) == 250, f"Not all samples were returned. Samples returned: {response.keys}"

    check_data(response.keys, FIRST_ADDED_CSVS)

    add_files_to_dataset(10, 20, [], SECOND_ADDED_CSVS)  # Add more samples to the dataset.

    for i in range(500):
        responses = list(get_new_data_since(CSV_UPDATED_TIME_STAMPS[9] + 1))
        assert len(responses) < 2, f"Received batched response, shouldn't happen: {responses}"
        if len(responses) == 1:
            response = responses[0]
            if len(response.keys) == 250:
                break
        time.sleep(1)

    assert response is not None, "Did not get any response from Storage"
    assert len(response.keys) == 250, f"Not all samples were returned. Samples returned: {response.keys}"

    check_data(response.keys, SECOND_ADDED_CSVS)

    responses = list(get_data_in_interval(0, CSV_UPDATED_TIME_STAMPS[9]))
    assert len(responses) == 1, f"Received batched/no response, shouldn't happen: {responses}"
    response = responses[0]

    check_data(response.keys, FIRST_ADDED_CSVS)

    check_get_current_timestamp()  # Check if the storage service is still available.


def main() -> None:
    try:
        test_storage()
    finally:
        cleanup_storage_database()
        cleanup_dataset_dir()


if __name__ == "__main__":
    main()
