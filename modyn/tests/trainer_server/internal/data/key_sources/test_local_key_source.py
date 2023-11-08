import math

import pytest
import torch
from modyn.trainer_server.internal.dataset.key_sources import LocalKeySource
from modyn.trainer_server.internal.dataset.local_dataset_writer import LocalDatasetWriter

TMP_PATH_TEST = "/tmp/offline_dataset_test"


def write_directory(pipeline_id, trigger_id, path, number_of_files, maximum_keys_in_memory):
    writer = LocalDatasetWriter(
        pipeline_id=pipeline_id,
        trigger_id=trigger_id,
        number_of_workers=1,
        maximum_keys_in_memory=maximum_keys_in_memory,
        offline_dataset_path=path,
    )
    # clean before writing
    writer.clean_this_trigger_samples()
    samples = prepare_samples(1, maximum_keys_in_memory * number_of_files)
    writer.inform_samples(*samples)


def prepare_samples(start_index: int, size: int):
    assert start_index > 0  # just to avoid division by zero errors
    tmp = list(range(start_index, start_index + size))
    weights = [0] * size
    for index in range(size):
        weights[index] = 1.0 / tmp[index]
    return tmp, torch.Tensor(weights)


def test_init():
    keysource = LocalKeySource(pipeline_id=12, trigger_id=1, offline_dataset_path=TMP_PATH_TEST)
    assert keysource.offline_dataset_path == TMP_PATH_TEST
    assert keysource._trigger_id == 1
    assert keysource._pipeline_id == 12
    assert keysource.uses_weights()
    assert keysource.get_num_data_partitions() == 0


def test_read():
    maximum_keys_in_memory = 25
    keysource = LocalKeySource(pipeline_id=12, trigger_id=1, offline_dataset_path=TMP_PATH_TEST)
    # to avoid wrong results from previously failed tests
    keysource.end_of_trigger_cleaning()
    write_directory(12, 1, TMP_PATH_TEST, number_of_files=4, maximum_keys_in_memory=maximum_keys_in_memory)
    assert keysource.get_num_data_partitions() == 4

    for i in range(keysource.get_num_data_partitions()):
        keys, weights = keysource.get_keys_and_weights(worker_id=0, partition_id=i)
        assert keys == list(range(1 + i * maximum_keys_in_memory, 1 + (i + 1) * maximum_keys_in_memory))
        assert all(math.isclose(k * v, 1, abs_tol=1e-5) for k, v in zip(keys, weights))

    # now add 2 files
    write_directory(12, 1, TMP_PATH_TEST, number_of_files=6, maximum_keys_in_memory=maximum_keys_in_memory)
    assert keysource.get_num_data_partitions() == 6

    for i in range(keysource.get_num_data_partitions()):
        keys, weights = keysource.get_keys_and_weights(worker_id=0, partition_id=i)
        assert keys == list(range(1 + i * maximum_keys_in_memory, 1 + (i + 1) * maximum_keys_in_memory))
        assert all(math.isclose(k * v, 1, abs_tol=1e-5) for k, v in zip(keys, weights))

    keysource.end_of_trigger_cleaning()
    assert keysource.get_num_data_partitions() == 0


def test_read_dirty_directory():
    # read from a directory that has samples belonging to another pipeline_id
    current_pipeline = 12
    other_pipeline = 99
    maximum_keys_in_memory = 25

    write_directory(other_pipeline, 1, TMP_PATH_TEST, number_of_files=10, maximum_keys_in_memory=maximum_keys_in_memory)

    keysource = LocalKeySource(pipeline_id=current_pipeline, trigger_id=1, offline_dataset_path=TMP_PATH_TEST)

    assert keysource.get_num_data_partitions() == 0
    with pytest.raises(ValueError):
        assert keysource.get_keys_and_weights(0, 0)

    write_directory(
        current_pipeline, 1, TMP_PATH_TEST, number_of_files=4, maximum_keys_in_memory=maximum_keys_in_memory
    )

    assert keysource.get_num_data_partitions() == 4

    for i in range(keysource.get_num_data_partitions()):
        keys, weights = keysource.get_keys_and_weights(worker_id=0, partition_id=i)
        assert keys == list(range(1 + i * maximum_keys_in_memory, 1 + (i + 1) * maximum_keys_in_memory))
        assert all(math.isclose(k * v, 1, abs_tol=1e-5) for k, v in zip(keys, weights))

    keysource.end_of_trigger_cleaning()

    # now check that the other pipeline wasn't affected
    ks_other = LocalKeySource(pipeline_id=other_pipeline, trigger_id=1, offline_dataset_path=TMP_PATH_TEST)
    assert ks_other.get_num_data_partitions() == 10

    for i in range(ks_other.get_num_data_partitions()):
        keys, weights = ks_other.get_keys_and_weights(worker_id=0, partition_id=i)
        assert keys == list(range(1 + i * maximum_keys_in_memory, 1 + (i + 1) * maximum_keys_in_memory))
        assert all(math.isclose(k * v, 1, abs_tol=1e-5) for k, v in zip(keys, weights))

    ks_other.end_of_trigger_cleaning()


def test_reads_pro():
    writer = LocalDatasetWriter(
        pipeline_id=0, trigger_id=0, number_of_workers=1, maximum_keys_in_memory=25, offline_dataset_path=TMP_PATH_TEST
    )
    reader = LocalKeySource(pipeline_id=0, trigger_id=0, offline_dataset_path=TMP_PATH_TEST)
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

    reader.end_of_trigger_cleaning()
