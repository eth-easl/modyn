import os
import pathlib
import shutil
import tempfile

import pytest
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.selector.internal.selector_strategies import CoresetStrategy
from modyn.selector.internal.selector_strategies.presampling_strategies.random_presampling_strategy import (
    RandomPresamplingStrategy,
)
from modyn.selector.internal.storage_backend.database.database_storage_backend import DatabaseStorageBackend

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"

TMP_DIR = tempfile.mkdtemp()


def get_config():
    return {
        "ratio": 50,
        "strategy": "Random",
    }


def get_minimal_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "hostname": "",
            "port": "0",
            "database": f"{database_path}",
        },
        "selector": {"insertion_threads": 8, "trigger_sample_directory": TMP_DIR},
    }


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield

    os.remove(database_path)
    shutil.rmtree(TMP_DIR)


def test_constructor():
    strat = RandomPresamplingStrategy(
        get_config(),
        get_minimal_modyn_config(),
        10,
        DatabaseStorageBackend(0, get_minimal_modyn_config(), 123),
    )
    assert strat.presampling_ratio == 50
    assert strat.requires_trigger_dataset_size


def test_target_size():
    strat = RandomPresamplingStrategy(
        get_config(),
        get_minimal_modyn_config(),
        10,
        DatabaseStorageBackend(0, get_minimal_modyn_config(), 123),
    )
    assert strat.get_target_size(100, None) == 50
    assert strat.get_target_size(20, None) == 10
    assert strat.get_target_size(19, None) == 9
    assert strat.get_target_size(0, None) == 0

    assert strat.get_target_size(100, 25) == 25
    assert strat.get_target_size(20, 12) == 10
    assert strat.get_target_size(19, 6) == 6
    assert strat.get_target_size(0, 121) == 0

    with pytest.raises(AssertionError):
        strat.get_target_size(-1, None)


def test_get_query_wrong():
    strat = RandomPresamplingStrategy(
        get_config(),
        get_minimal_modyn_config(),
        10,
        DatabaseStorageBackend(0, get_minimal_modyn_config(), 123),
    )

    # missing size
    with pytest.raises(AssertionError):
        strat.get_presampling_query(120, None, None, None)

    # negative size
    with pytest.raises(AssertionError):
        strat.get_presampling_query(120, None, None, -1)

    # negative limit
    with pytest.raises(AssertionError):
        strat.get_presampling_query(120, None, -18, 120)


def test_constructor_throws_on_invalid_config():
    conf = get_config()
    conf["ratio"] = 0

    with pytest.raises(ValueError):
        RandomPresamplingStrategy(
            conf,
            get_minimal_modyn_config(),
            10,
            DatabaseStorageBackend(0, get_minimal_modyn_config(), 123),
        )

    conf["ratio"] = 101

    with pytest.raises(ValueError):
        RandomPresamplingStrategy(
            conf,
            get_minimal_modyn_config(),
            10,
            DatabaseStorageBackend(0, get_minimal_modyn_config(), 123),
        )


def test_dataset_size_various_scenarios():
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels1 = [0] * 10

    data2 = list(range(10, 40))
    timestamps2 = list(range(10, 40))
    labels2 = [0] * 30

    conf = get_config()
    conf["limit"] = -1
    conf["reset_after_trigger"] = True

    # first trigger
    strat = CoresetStrategy(
        {"presampling_config": conf, "limit": -1, "reset_after_trigger": True},
        get_minimal_modyn_config(),
        0,
        100,
    )
    presampling_strat: RandomPresamplingStrategy = strat.presampling_strategy
    strat.inform_data(data1, timestamps1, labels1)
    trigger_size = strat._get_trigger_dataset_size()
    assert presampling_strat.get_target_size(trigger_size, None) == 5  # 50% of presampling
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 5
    assert trigger_num_partitions == 1

    # second trigger
    strat.inform_data(data2, timestamps2, labels2)
    trigger_size = strat._get_trigger_dataset_size()
    assert presampling_strat.get_target_size(trigger_size, None) == 15  # 50% of presampling

    # limited capacity
    strat.has_limit = True
    strat.training_set_size_limit = 10
    trigger_size = strat._get_trigger_dataset_size()
    assert presampling_strat.get_target_size(trigger_size, None) == 15  # 50% of presampling

    # only trigger data
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert all(int(key) >= 10 for (key, _) in strat.get_trigger_partition_keys(trigger_id, 0))

    # remove the trigger
    strat.reset_after_trigger = False
    strat.tail_triggers = None
    trigger_size = strat._get_trigger_dataset_size()
    assert presampling_strat.get_target_size(trigger_size, None) == 20  # 50% of presampling

    # remove the limit
    strat._has_limit = False
    trigger_size = strat._get_trigger_dataset_size()
    assert presampling_strat.get_target_size(trigger_size, None) == 20  # 50% of presampling

    # adjust the presampling
    presampling_strat.presampling_ratio = 75
    trigger_size = strat._get_trigger_dataset_size()
    assert presampling_strat.get_target_size(trigger_size, None) == 30  # 75% of presampling

    # set tail triggering
    strat.tail_triggers = 1
    trigger_size = strat._get_trigger_dataset_size()
    assert presampling_strat.get_target_size(trigger_size, None) == 22  # 75% of presampling
