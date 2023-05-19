import math
import os
import shutil

import numpy as np
import pytest
from modyn.trainer_server.internal.dataset.local_dataset_handler import LocalDatasetHandler


def clean_directory():
    if ".tmp_offline_dataset" in os.listdir():
        shutil.rmtree(".tmp_offline_dataset")


def prepare_samples(start_index: int, size: int):
    assert start_index > 0  # just to avoid division by zero errors
    tmp = list(range(start_index, start_index + size))
    array = np.empty(size, dtype=np.dtype("i8,f8"))
    for index in range(size):
        array[index] = (tmp[index], 1.0 / tmp[index])
    return array


def test_init():
    clean_directory()
    handler = LocalDatasetHandler(12, 1, 1, 25)
    assert ".tmp_offline_dataset" in os.listdir()
    assert len(os.listdir(".tmp_offline_dataset")) == 0
    assert handler.current_file_index == 0
    assert handler.current_sample_index == 0
    assert handler.maximum_keys_in_memory == 25


def test_init_just_read():
    clean_directory()
    handler = LocalDatasetHandler(12, 1, 1)
    assert ".tmp_offline_dataset" in os.listdir()
    assert len(os.listdir(".tmp_offline_dataset")) == 0
    assert handler.current_file_index == 0
    assert handler.current_sample_index == 0
    assert handler.maximum_keys_in_memory is None

    with pytest.raises(AssertionError):
        handler.inform_samples(np.array([11]))
    with pytest.raises(AssertionError):
        handler.store_last_samples()


def test_writes():
    clean_directory()
    handler = LocalDatasetHandler(12, 1, 1, 25)

    assert ".tmp_offline_dataset" in os.listdir()
    assert len(os.listdir(".tmp_offline_dataset")) == 0

    handler.inform_samples(prepare_samples(1, 50))
    assert len(os.listdir(".tmp_offline_dataset")) == 2

    handler.inform_samples(prepare_samples(1, 49))
    assert len(os.listdir(".tmp_offline_dataset")) == 3

    handler.store_last_samples()
    assert len(os.listdir(".tmp_offline_dataset")) == 4

    handler.clean_this_trigger_samples()
    assert len(os.listdir(".tmp_offline_dataset")) == 0


def test_reads_pro():
    clean_directory()
    handler = LocalDatasetHandler(pipeline_id=0, trigger_id=0, number_of_workers=2, maximum_keys_in_memory=25)
    # We can just keep 25 keys in memory. So, after it we dup to 2 files (one for each worker)

    # 50 samples: we expect 4 files 1) p=0, w=0 containing [1,12], 2) p=0, w=1 containing [13,25]
    # 3) p=1, w=0 containing [26,37], 4) p=1, w=1 containing [38,50]
    samples = prepare_samples(1, 50)
    handler.inform_samples(samples)

    assert handler.get_number_of_partitions() == 2

    k00, w00 = handler.get_keys_and_weights(0, 0)
    assert k00 == list(range(1, 13))
    assert all(math.isclose(k * v, 1) for k, v in zip(k00, w00))  # get the correct key

    k01, w01 = handler.get_keys_and_weights(0, 1)
    assert k01 == list(range(13, 26))
    assert all(math.isclose(k * v, 1) for k, v in zip(k01, w01))  # get the correct key

    k10, w10 = handler.get_keys_and_weights(1, 0)
    assert k10 == list(range(26, 38))
    assert all(math.isclose(k * v, 1) for k, v in zip(k10, w10))  # get the correct key

    k11, w11 = handler.get_keys_and_weights(1, 1)
    assert k11 == list(range(38, 51))
    assert all(math.isclose(k * v, 1) for k, v in zip(k11, w11))  # get the correct key

    assert k00 + k01 + k10 + k11 == [key for key, _ in prepare_samples(1, 50)]
    assert w00 + w01 + w10 + w11 == [weight for _, weight in prepare_samples(1, 50)]

    # 3 samples: we expect 4 files (same as before, so 2 partitions) and then 6 when we force the flush.
    samples = prepare_samples(1000, 4)
    handler.inform_samples(samples)

    assert handler.get_number_of_partitions() == 2
    handler.store_last_samples()
    assert handler.get_number_of_partitions() == 3

    # check that everything is ok
    k20, w20 = handler.get_keys_and_weights(2, 0)
    assert k20 == list(range(1000, 1002))
    assert all(math.isclose(k * v, 1) for k, v in zip(k20, w20))  # get the correct key

    k21, w21 = handler.get_keys_and_weights(2, 1)
    assert k21 == list(range(1002, 1004))
    assert all(math.isclose(k * v, 1) for k, v in zip(k21, w21))  # get the correct key
