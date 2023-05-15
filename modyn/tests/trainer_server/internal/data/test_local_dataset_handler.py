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
    handler = LocalDatasetHandler(12, 25)
    assert ".tmp_offline_dataset" in os.listdir()
    assert len(os.listdir(".tmp_offline_dataset")) == 0
    assert handler.current_file_index == 0
    assert handler.current_sample_index == 0
    assert handler.file_size == 25


def test_init_just_read():
    clean_directory()
    handler = LocalDatasetHandler(12)
    assert ".tmp_offline_dataset" in os.listdir()
    assert len(os.listdir(".tmp_offline_dataset")) == 0
    assert handler.current_file_index == 0
    assert handler.current_sample_index == 0
    assert handler.file_size == -1

    with pytest.raises(AssertionError):
        handler.inform_samples(np.array([11]))
    with pytest.raises(AssertionError):
        handler.store_last_samples()


def test_writes():
    clean_directory()
    handler = LocalDatasetHandler(12, 25)

    assert ".tmp_offline_dataset" in os.listdir()
    assert len(os.listdir(".tmp_offline_dataset")) == 0

    handler.inform_samples(prepare_samples(1, 50))
    assert len(os.listdir(".tmp_offline_dataset")) == 2

    handler.inform_samples(prepare_samples(1, 49))
    assert len(os.listdir(".tmp_offline_dataset")) == 3

    handler.store_last_samples()
    assert len(os.listdir(".tmp_offline_dataset")) == 4

    handler.clean_working_directory()
    assert ".tmp_offline_dataset" not in os.listdir()


def test_reads():
    clean_directory()
    handler = LocalDatasetHandler(12, 25)
    assert ".tmp_offline_dataset" in os.listdir()
    assert len(os.listdir(".tmp_offline_dataset")) == 0

    handler.inform_samples(prepare_samples(1, 50))
    assert len(os.listdir(".tmp_offline_dataset")) == 2

    keys, weights = handler.get_keys_and_weights(0, 0)

    for key, weight in zip(keys, weights):
        assert key == 1 + keys.index(key)
        assert weight == 1.0 / (1 + weights.index(weight))
        assert math.isclose(key * weight, 1)

    keys, weights = handler.get_keys_and_weights(0, 1)

    for key, weight in zip(keys, weights):
        assert key == 26 + keys.index(key)
        assert weight == 1.0 / (26 + weights.index(weight))
        assert math.isclose(key * weight, 1)

    handler.inform_samples(prepare_samples(1000, 49))
    assert len(os.listdir(".tmp_offline_dataset")) == 3

    handler.store_last_samples()
    assert len(os.listdir(".tmp_offline_dataset")) == 4

    # redo the same check
    keys, weights = handler.get_keys_and_weights(0, 0)

    for key, weight in zip(keys, weights):
        assert key == 1 + keys.index(key)
        assert weight == 1.0 / (1 + weights.index(weight))
        assert math.isclose(key * weight, 1)

    # check the new file

    keys, weights = handler.get_keys_and_weights(0, 2)

    for key, weight in zip(keys, weights):
        assert key == 1000 + keys.index(key)
        assert weight == 1.0 / (1000 + weights.index(weight))
        assert math.isclose(key * weight, 1)

    keys, weights = handler.get_keys_and_weights(0, 3)

    for key, weight in zip(keys, weights):
        assert key == 1025 + keys.index(key)
        assert weight == 1.0 / (1025 + weights.index(weight))
        assert math.isclose(key * weight, 1)
