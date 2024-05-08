# pylint: disable=not-callable, unnecessary-comprehension
import os
import pathlib
import shutil
import tempfile

import pytest
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.selector.internal.selector_strategies.coreset_strategy import CoresetStrategy
from modyn.selector.internal.selector_strategies.presampling_strategies import LabelBalancedPresamplingStrategy

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
    return {
        "reset_after_trigger": False,
        "presampling_config": {"ratio": 50, "strategy": "LabelBalanced"},
        "limit": -1,
    }


def insert_data(strat, base_index=0):
    strat.inform_data(
        range(base_index, base_index + 150),
        range(base_index, base_index + 150),
        [0] * 100 + [1] * 25 + [2] * 20 + [3] * 5,
    )


def insert_data_clustered(strat, per_class):
    for index, number in enumerate(per_class):
        strat.inform_data(
            range(1000 * index, 1000 * index + number),
            range(1000 * index, 1000 * index + number),
            [index] * number,
        )


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
    strat = CoresetStrategy(get_config(), modyn_config, 0, 1000)

    strat.presampling_strategy: LabelBalancedPresamplingStrategy

    insert_data(strat, base_index=0)

    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [100, 25, 20, 5]

    insert_data(strat, base_index=200)

    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [200, 50, 40, 10]

    strat.inform_data([1000], [1000], [67])
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [200, 50, 40, 10, 1]

    strat.inform_data([1100], [1000], [67])
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [200, 50, 40, 10, 2]

    strat.inform_data([1200], [1000], [69])
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [200, 50, 40, 10, 2, 1]


def test_counting_query_reset_after_trigger():
    modyn_config = get_minimal_modyn_config()
    strat = CoresetStrategy(get_config(), modyn_config, 0, 1000)

    strat.presampling_strategy: LabelBalancedPresamplingStrategy

    insert_data(strat, base_index=0)

    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, 0)
    assert count == [100, 25, 20, 5]
    strat.trigger()

    # new trigger, reset the counters
    insert_data(strat, base_index=200)

    # query with reset
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, 0)
    assert count == [100, 25, 20, 5]

    # query without reset
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, None)
    assert count == [200, 50, 40, 10]

    strat.inform_data([1000], [1000], [67])
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, 0)
    assert count == [100, 25, 20, 5, 1]

    strat.inform_data([1100], [1000], [67])
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, 0)
    assert count == [100, 25, 20, 5, 2]

    strat.inform_data([1200], [1000], [69])
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, 0)
    assert count == [100, 25, 20, 5, 2, 1]

    strat.trigger()
    strat.inform_data([1300], [1000], [69])
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, 0)
    assert count == [1]

    # now use tail trigger = 1 (so it should only consider the last two triggers)
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, 1)
    assert count == [100, 25, 20, 5, 2, 2]

    # now use tail trigger = 2 (so it should consider all the triggers)
    count = strat.presampling_strategy._get_samples_count_per_balanced_column(strat._next_trigger_id, 2)
    assert count == [200, 50, 40, 10, 2, 2]


def test_query_data_above_threshold():
    modyn_config = get_minimal_modyn_config()
    strat = CoresetStrategy(get_config(), modyn_config, 0, 1000)

    strat.presampling_strategy: LabelBalancedPresamplingStrategy

    insert_data_clustered(strat, [15, 15, 30])

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 60

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 30

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) == 10


def test_query_data_one_below_threshold():
    modyn_config = get_minimal_modyn_config()
    strat = CoresetStrategy(get_config(), modyn_config, 0, 1000)

    strat.presampling_strategy: LabelBalancedPresamplingStrategy
    count = [40, 160, 100]
    insert_data_clustered(strat, count)

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 300

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 150
    # 40 class0, 55 class1 and 2

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) == min(count[index], 55)


def test_query_data_one_below_threshold_balanced():
    modyn_config = get_minimal_modyn_config()
    config = get_config()
    config["presampling_config"]["force_column_balancing"] = True
    strat = CoresetStrategy(config, modyn_config, 0, 1000)

    strat.presampling_strategy: LabelBalancedPresamplingStrategy
    count = [40, 160, 100]
    insert_data_clustered(strat, count)

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 300

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 150
    # 40 for each class

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) == 40


def test_query_data_below_target():
    modyn_config = get_minimal_modyn_config()
    config = get_config()
    config["presampling_config"]["ratio"] = 72  # to get 100 points as target
    strat = CoresetStrategy(config, modyn_config, 0, 1000)

    strat.presampling_strategy: LabelBalancedPresamplingStrategy
    count = [30, 50, 40, 20]
    insert_data_clustered(strat, count)

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 140

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 100
    # 40 for each class

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) == min(26, count[index])

    assert len(selected) == 98


def test_query_data_below_target_forced():
    modyn_config = get_minimal_modyn_config()
    config = get_config()
    config["presampling_config"]["ratio"] = 72  # to get 100 points as target
    config["presampling_config"]["force_required_target_size"] = True
    strat = CoresetStrategy(config, modyn_config, 0, 1000)

    strat.presampling_strategy: LabelBalancedPresamplingStrategy
    count = [30, 50, 40, 20]
    insert_data_clustered(strat, count)

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 140

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 100
    # 40 for each class

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) <= min(26 + 1, count[index])  # you can get up to one sample more than the FS

    assert len(selected) == 100  # but the total is the target size


def test_query_data_one_below_threshold_force_size_but_already_ok():
    modyn_config = get_minimal_modyn_config()
    config = get_config()
    config["presampling_config"]["force_required_target_size"] = True
    strat = CoresetStrategy(config, modyn_config, 0, 1000)

    strat.presampling_strategy: LabelBalancedPresamplingStrategy
    count = [40, 160, 100]
    insert_data_clustered(strat, count)

    this_trigger_size = strat._get_trigger_dataset_size()
    assert this_trigger_size == 300

    assert strat.presampling_strategy.get_target_size(this_trigger_size, None) == 150
    # 40 class0, 55 class1 and 2

    selected = list(strat._get_data())[0]

    for index in range(3):
        label_index = [el for el in selected if index * 1000 <= el < (index + 1) * 1000]
        assert len(label_index) == min(count[index], 55)
