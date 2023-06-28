import os

import torch
from modyn.trainer_server.internal.dataset.local_dataset_writer import LocalDatasetWriter

OFFLINE_DATASET_PATH = "/tmp/offline_dataset"


def prepare_samples(start_index: int, size: int):
    assert start_index > 0  # just to avoid division by zero errors
    tmp = list(range(start_index, start_index + size))
    weights = [0] * size
    for index in range(size):
        weights[index] = 1.0 / tmp[index]
    return tmp, torch.Tensor(weights)


def test_init_writer():
    maximum_keys_in_memory = 25
    pipeline_id = 12
    trigger_id = 198
    number_of_workers = 1

    writer = LocalDatasetWriter(
        pipeline_id, trigger_id, number_of_workers, maximum_keys_in_memory, OFFLINE_DATASET_PATH
    )
    assert "offline_dataset" in os.listdir("/tmp/")
    assert len(os.listdir(OFFLINE_DATASET_PATH)) == 0
    assert writer.current_file_index == 0
    assert writer.current_sample_index == 0

    assert writer.maximum_keys_in_memory == maximum_keys_in_memory
    assert writer.pipeline_id == pipeline_id
    assert writer.trigger_id == trigger_id
    assert writer.number_of_workers == number_of_workers
    assert len(writer.output_samples_list) == maximum_keys_in_memory

    writer.clean_this_trigger_samples()


def test_writes():
    writer = LocalDatasetWriter(12, 1, 1, 25, OFFLINE_DATASET_PATH)

    assert writer.current_file_index == 0
    assert "offline_dataset" in os.listdir("/tmp/")
    assert len(os.listdir(OFFLINE_DATASET_PATH)) == 0
    assert writer.maximum_keys_in_memory == 25
    assert writer.get_number_of_partitions() == 0

    # 50 samples
    writer.inform_samples(*prepare_samples(1, 50))
    assert len(os.listdir(OFFLINE_DATASET_PATH)) == 2
    assert writer.current_file_index == 2
    assert writer.current_sample_index == 0
    assert writer.get_number_of_partitions() == 2

    # 99 samples
    writer.inform_samples(*prepare_samples(1, 49))
    assert len(os.listdir(OFFLINE_DATASET_PATH)) == 3
    assert writer.current_file_index == 3
    assert writer.current_sample_index == 24
    assert writer.get_number_of_partitions() == 3

    # write the last sample
    writer.finalize()
    assert len(os.listdir(OFFLINE_DATASET_PATH)) == 4
    assert writer.current_file_index == 4
    assert writer.current_sample_index == 0
    assert writer.get_number_of_partitions() == 4

    # check that the last file is smaller (since 24/25 samples)
    full_file = os.path.join(OFFLINE_DATASET_PATH, "12_1_0_0.npy")
    last_file = os.path.join(OFFLINE_DATASET_PATH, "12_1_3_0.npy")
    another_full_file = os.path.join(OFFLINE_DATASET_PATH, "12_1_2_0.npy")

    assert os.path.getsize(full_file) == os.path.getsize(another_full_file) and os.path.getsize(
        full_file
    ) > os.path.getsize(last_file)

    # finalize on 0 samples, should not increment the file_index
    writer.finalize()
    assert writer.current_file_index == 4

    writer.clean_this_trigger_samples()
    assert len(os.listdir(OFFLINE_DATASET_PATH)) == 0
    assert writer.current_file_index == 4
    assert writer.get_number_of_partitions() == 0
