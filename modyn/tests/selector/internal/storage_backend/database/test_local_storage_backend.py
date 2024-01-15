import os
from pathlib import Path
import shutil
import tempfile

import pytest
from modyn.selector.internal.storage_backend.local import LocalStorageBackend
from modyn.utils.utils import flatten

TMP_DIR_TSAMPLES = tempfile.mkdtemp()
TMP_DIR_STORAGE = tempfile.mkdtemp()


def get_minimal_modyn_config():
    return {
        "metadata_database": {"drivername": "sqlite", "username": "", "password": "", "host": "", "port": "0"},
        "selector": {
            "insertion_threads": 8,
            "trigger_sample_directory": TMP_DIR_TSAMPLES,
            "local_storage_directory": TMP_DIR_STORAGE,
        },
    }


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    Path(TMP_DIR_TSAMPLES).mkdir(parents=True, exist_ok=True)
    Path(TMP_DIR_STORAGE).mkdir(parents=True, exist_ok=True)

    yield
    shutil.rmtree(TMP_DIR_TSAMPLES)
    shutil.rmtree(TMP_DIR_STORAGE)


def test_get_data_since_trigger():
    backend = LocalStorageBackend(42, get_minimal_modyn_config(), 2)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])
    backend.persist_samples(1, [13, 14, 15], [3, 4, 5], [40, 41, 42])
    backend.persist_samples(2, [16, 17, 18], [6, 7, 8], [40, 41, 42])

    data_since_trigger_one = [keys for keys, _ in backend.get_data_since_trigger(1)]
    assert len(data_since_trigger_one) == 3  # Validate partitioning
    assert set(flatten(data_since_trigger_one)) == set([13, 14, 15, 16, 17, 18])  # Validate content


def test_get_trigger_data():
    backend = LocalStorageBackend(42, get_minimal_modyn_config(), 2)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])
    backend.persist_samples(1, [13, 14, 15], [3, 4, 5], [40, 41, 42])
    backend.persist_samples(2, [16, 17, 18], [6, 7, 8], [40, 41, 42])

    trigger_zero_data = [keys for keys, _ in backend.get_trigger_data(0)]
    trigger_one_data = [keys for keys, _ in backend.get_trigger_data(1)]
    trigger_three_data = [keys for keys, _ in backend.get_trigger_data(3)]

    # TODO: change tests to set equality as order does not matter
    assert len(trigger_zero_data) == 2  # Validate partitioning
    assert set(flatten(trigger_zero_data)) == set([10, 11, 12])  # Validate content

    assert len(trigger_one_data) == 2  # Validate partitioning
    assert set(flatten(trigger_one_data)) == set([13, 14, 15])  # Validate content

    assert len(trigger_three_data) == 0


def test_get_all_data():
    backend = LocalStorageBackend(42, get_minimal_modyn_config(), 2)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])
    backend.persist_samples(1, [13, 14, 15], [3, 4, 5], [40, 41, 42])
    backend.persist_samples(2, [16, 17, 18], [6, 7, 8], [40, 41, 42])

    all_data = [keys for keys, _ in backend.get_all_data()]

    assert len(all_data) == 5  # Validate partitioning
    assert set(flatten(all_data)) == set([10, 11, 12, 13, 14, 15, 16, 17, 18])  # Validate content


def test_get_many_data():
    N = 10000
    backend = LocalStorageBackend(42, get_minimal_modyn_config(), N)

    sample_count = 1500000
    samples = list(range(sample_count))
    backend.persist_samples(0, samples, [0] * sample_count, [10] * sample_count)

    all_data = [keys for keys, _ in backend.get_all_data()]

    assert len(all_data) == sample_count // N  # Validate partitioning
    assert set(flatten(all_data)) == set(samples)  # Validate content


def test_get_available_labels_with_tail():
    backend = LocalStorageBackend(1, get_minimal_modyn_config(), 42)
    # Next trigger is 0
    assert sorted(backend.get_available_labels(0)) == []

    backend.persist_samples(0, [0, 1, 2, 3], [0, 0, 0, 0], [1, 18, 1, 0])

    # Next trigger is 1, trigger 0 has data now
    assert sorted(backend.get_available_labels(1)) == [0, 1, 18]

    backend.persist_samples(1, [4, 5], [0, 0, 0, 0], [0, 890])

    # Next trigger is 2, trigger 1 has data now
    assert sorted(backend.get_available_labels(2, tail_triggers=0)) == [0, 890]


def test_get_available_labels_no_tail():
    backend = LocalStorageBackend(1, get_minimal_modyn_config(), 42)
    # Next trigger is 0
    assert sorted(backend.get_available_labels(0)) == []

    backend.persist_samples(0, [0, 1, 2, 3], [0, 0, 0, 0], [1, 18, 1, 0])

    # Next trigger is 1, trigger 0 has data now
    assert sorted(backend.get_available_labels(1)) == [0, 1, 18]

    backend.persist_samples(1, [4, 5], [0, 0, 0, 0], [0, 890])

    # Next trigger is 2, trigger 1 has data now
    assert sorted(backend.get_available_labels(2)) == [0, 1, 18, 890]


# BEGIN TESTS OF INTERNAL FUNCTIONS #


def test_get_all_data_yield_per():
    backend = LocalStorageBackend(42, get_minimal_modyn_config(), 2)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])

    backend._maximum_keys_in_memory = 1
    all_data_yp1 = [keys for keys, _ in backend.get_all_data()]
    backend._maximum_keys_in_memory = 2
    all_data_yp2 = [keys for keys, _ in backend.get_all_data()]
    backend._maximum_keys_in_memory = 3
    all_data_yp3 = [keys for keys, _ in backend.get_all_data()]

    assert set(flatten(all_data_yp1)) == set(flatten(all_data_yp2))
    assert set(flatten(all_data_yp2)) == set(flatten(all_data_yp3))
    assert set(flatten(all_data_yp3)) == set([10, 11, 12])

    assert len(all_data_yp1) == 3
    assert len(all_data_yp2) == 2
    assert len(all_data_yp3) == 1
