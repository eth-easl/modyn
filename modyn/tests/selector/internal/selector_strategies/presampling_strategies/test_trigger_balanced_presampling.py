# pylint: disable=not-callable, unnecessary-comprehension
import os
import pathlib
import shutil
import tempfile

import pytest
from modyn.config.schema.pipeline import CoresetStrategyConfig, PresamplingConfig
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.selector.internal.selector_strategies.coreset_strategy import CoresetStrategy
from modyn.selector.internal.selector_strategies.presampling_strategies import TriggerBalancedPresamplingStrategy

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"

TMP_DIR = tempfile.mkdtemp()


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


def get_config():
    return CoresetStrategyConfig(
        tail_triggers=None,
        presampling_config=PresamplingConfig(ratio=50, strategy="TriggerBalanced"),
        limit=-1,
        maximum_keys_in_memory=1000,
    )


def insert_data(strat, base_index=0, amount=150):
    strat.inform_data(
        range(base_index, base_index + amount),
        range(base_index, base_index + amount),
        [0] * amount,  # labels are useless
    )


def insert_data_several_triggers(strat, trigger_sizes):
    for i, trigger_size in enumerate(trigger_sizes):
        assert trigger_size < 1000
        insert_data(strat, base_index=i * 1000, amount=trigger_size)
        strat.trigger()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield

    os.remove(database_path)
    shutil.rmtree(TMP_DIR)


def test_counting_query():
    modyn_config = get_minimal_modyn_config()
    strat = CoresetStrategy(get_config(), modyn_config, 0)

    strat.presampling_strategy: TriggerBalancedPresamplingStrategy

    insert_data(strat, base_index=0, amount=150)

    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [150]

    strat.trigger()
    insert_data(strat, base_index=200, amount=120)

    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [150, 120]

    strat.inform_data([1000], [1000], [67])
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [150, 121]

    strat.trigger()
    strat.inform_data([1100], [1000], [67])
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [150, 121, 1]

    strat.inform_data([1200], [1000], [69])
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [150, 121, 2]


def test_query_data_above_threshold():
    modyn_config = get_minimal_modyn_config()
    strat = CoresetStrategy(get_config(), modyn_config, 0)

    strat.presampling_strategy: TriggerBalancedPresamplingStrategy

    insert_data_several_triggers(strat, [100, 120, 40, 60])
    assert strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None) == [
        100,
        120,
        40,
        60,
    ]

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 320

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 160

    selected = list(strat._get_data())[0]

    # check that we get 40 for each trigger
    for index in range(4):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) == 40


def test_query_data_one_below_threshold():
    modyn_config = get_minimal_modyn_config()
    strat = CoresetStrategy(get_config(), modyn_config, 0)

    strat.presampling_strategy: TriggerBalancedPresamplingStrategy
    count = [40, 160, 100]
    insert_data_several_triggers(strat, count)

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 300

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 150
    # 40 trigger0, 55 trigger1 and 2

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) == min(count[index], 55)


def test_query_data_one_below_threshold_balanced():
    modyn_config = get_minimal_modyn_config()
    config = get_config()
    config.presampling_config.force_column_balancing = True
    strat = CoresetStrategy(config, modyn_config, 0)

    strat.presampling_strategy: TriggerBalancedPresamplingStrategy
    count = [40, 160, 100]
    insert_data_several_triggers(strat, count)

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 300

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 150
    # 40 for each trigger

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) == 40


def test_query_data_below_target():
    modyn_config = get_minimal_modyn_config()
    config = get_config()
    config.presampling_config.ratio = 72  # to get 100 points as target
    strat = CoresetStrategy(config, modyn_config, 0)

    strat.presampling_strategy: TriggerBalancedPresamplingStrategy
    count = [30, 50, 40, 20]
    insert_data_several_triggers(strat, count)

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 140

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 100
    # 40 for each trigegr

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) == min(26, count[index])

    assert len(selected) == 98


def test_query_data_below_target_forced():
    modyn_config = get_minimal_modyn_config()
    config = get_config()
    config.presampling_config.ratio = 72  # to get 100 points as target
    config.presampling_config.force_required_target_size = True
    strat = CoresetStrategy(config, modyn_config, 0)

    strat.presampling_strategy: TriggerBalancedPresamplingStrategy
    count = [30, 50, 40, 20]
    insert_data_several_triggers(strat, count)

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 140

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 100
    # 40 for each trigger

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) <= min(26 + 1, count[index])  # you can get up to one sample more than the FS

    assert len(selected) == 100  # but the total is the target size


def test_query_data_one_below_threshold_force_size_but_already_ok():
    modyn_config = get_minimal_modyn_config()
    config = get_config()
    config.presampling_config.force_required_target_size = True
    strat = CoresetStrategy(config, modyn_config, 0)

    strat.presampling_strategy: TriggerBalancedPresamplingStrategy
    count = [40, 160, 100]
    insert_data_several_triggers(strat, count)

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 300

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 150
    # 40 trigger0, 55 trigger1 and 2

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) == min(count[index], 55)
