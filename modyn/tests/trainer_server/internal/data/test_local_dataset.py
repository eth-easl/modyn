import math
import os
import shutil

import torch
from modyn.trainer_server.internal.dataset.key_sources import LocalKeySource
from modyn.trainer_server.internal.dataset.local_dataset_writer import LocalDatasetWriter

OFFLINE_DATASET_PATH = "/tmp/offline_dataset"


def clean_directory():
    if "offline_dataset" in os.listdir("/tmp"):
        shutil.rmtree("/tmp/offline_dataset")


def prepare_samples(start_index: int, size: int):
    assert start_index > 0  # just to avoid division by zero errors
    tmp = list(range(start_index, start_index + size))
    weights = [0] * size
    for index in range(size):
        weights[index] = 1.0 / tmp[index]
    return tmp, torch.Tensor(weights)


def test_init_writer():
    clean_directory()
    handler = LocalDatasetWriter(12, 1, 1, 25, OFFLINE_DATASET_PATH)
    assert "offline_dataset" in os.listdir("/tmp/")
    assert len(os.listdir("/tmp/offline_dataset")) == 0
    assert handler.current_file_index == 0
    assert handler.current_sample_index == 0
    assert handler.maximum_keys_in_memory == 25
    clean_directory()


def test_init_just_read():
    clean_directory()
    handler = LocalKeySource(12, 1, OFFLINE_DATASET_PATH)
    assert "offline_dataset" in os.listdir("/tmp/")
    assert len(os.listdir("/tmp/offline_dataset")) == 0
    assert handler._pipeline_id == 12
    assert handler._trigger_id == 1
    clean_directory()


def test_writes():
    clean_directory()
    handler = LocalDatasetWriter(12, 1, 1, 25, OFFLINE_DATASET_PATH)

    assert "offline_dataset" in os.listdir("/tmp/")
    assert len(os.listdir("/tmp/offline_dataset")) == 0
    assert handler.maximum_keys_in_memory == 25

    handler.inform_samples(*prepare_samples(1, 50))
    assert len(os.listdir("/tmp/offline_dataset")) == 2

    handler.inform_samples(*prepare_samples(1, 49))
    assert len(os.listdir("/tmp/offline_dataset")) == 3

    handler.finalize()
    assert len(os.listdir("/tmp/offline_dataset")) == 4

    handler.clean_this_trigger_samples()
    assert len(os.listdir("/tmp/offline_dataset")) == 0

    clean_directory()


def test_reads_pro():
    clean_directory()
    writer = LocalDatasetWriter(
        pipeline_id=0,
        trigger_id=0,
        number_of_workers=1,
        maximum_keys_in_memory=25,
        offline_dataset_path=OFFLINE_DATASET_PATH,
    )
    reader = LocalKeySource(pipeline_id=0, trigger_id=0, offline_dataset_path=OFFLINE_DATASET_PATH)
    # We can just keep 25 keys in memory. So, after it we dup to 2 files (one for each worker)

    # 50 samples: we expect 4 files 1) p=0, w=0 containing [1,12], 2) p=0, w=1 containing [13,25]
    # 3) p=1, w=0 containing [26,37], 4) p=1, w=1 containing [38,50]
    samples = prepare_samples(1, 50)
    writer.inform_samples(*samples)

    assert writer.get_number_of_partitions() == 2

    k00, w00 = reader.get_keys_and_weights(worker_id=0, partition_id=0)
    assert k00 == list(range(1, 26))
    assert all(math.isclose(k * v, 1, abs_tol=1e-5) for k, v in zip(k00, w00))  # get the correct key

    k10, w10 = reader.get_keys_and_weights(worker_id=0, partition_id=1)
    assert k10 == list(range(26, 51))
    assert all(math.isclose(k * v, 1, abs_tol=1e-5) for k, v in zip(k10, w10))  # get the correct key

    assert k00 + k10 == prepare_samples(1, 50)[0]
    assert w00 + w10 == prepare_samples(1, 50)[1].tolist()

    # 3 samples: we expect 4 files (same as before, so 2 partitions) and then 6 when we force the flush.
    samples = prepare_samples(1000, 4)
    writer.inform_samples(*samples)

    assert writer.get_number_of_partitions() == 2
    assert reader.get_num_data_partitions() == 2
    writer.finalize()
    assert writer.get_number_of_partitions() == 3
    assert reader.get_num_data_partitions() == 3

    # check that everything is ok
    k20, w20 = reader.get_keys_and_weights(worker_id=0, partition_id=2)
    assert k20 == list(range(1000, 1004))
    assert all(math.isclose(k * v, 1, abs_tol=1e-5) for k, v in zip(k20, w20))  # get the correct key

    clean_directory()
