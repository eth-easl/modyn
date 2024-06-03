# pylint: disable=no-value-for-parameter,redefined-outer-name,singleton-comparison
import os
import pathlib
import shutil
import tempfile
from math import isclose
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from modyn.config.schema.pipeline import NewDataStrategyConfig
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.new_data_strategy import NewDataStrategy
from modyn.utils import flatten

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
        "selector": {
            "insertion_threads": 8,
            "trigger_sample_directory": TMP_DIR,
        },
    }


def get_config():
    # TODO(MaxiBoether): also test local
    return NewDataStrategyConfig(tail_triggers=None, limit=-1, storage_backend="database", maximum_keys_in_memory=1000)


def get_config_tail():
    # TODO(MaxiBoether): also test local
    return NewDataStrategyConfig(tail_triggers=1, limit=-1, storage_backend="database", maximum_keys_in_memory=1000)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

    yield

    os.remove(database_path)
    shutil.rmtree(TMP_DIR)


def test_e2e_noreset_nolimit():
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels1 = [0] * 10

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))
    labels2 = [0] * 10

    conf = get_config()
    conf.maximum_keys_in_memory = 100
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)

    assert trigger_num_keys == 10
    assert trigger_num_partitions == 1
    assert {int(key) for (key, _) in training_samples} == set(range(10))

    strat.inform_data(data2, timestamps2, labels2)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)
    assert trigger_num_keys == 20
    assert trigger_num_partitions == 1
    assert {int(key) for (key, _) in training_samples} == set(range(20))


def test_e2e_noreset_nolimit_memory_limits():
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels1 = [0] * 10

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))
    labels2 = [0] * 10

    conf = get_config()
    conf.maximum_keys_in_memory = 5
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 10
    assert trigger_num_partitions == 2

    training_samples_part0 = strat.get_trigger_partition_keys(trigger_id, 0)
    training_samples_part1 = strat.get_trigger_partition_keys(trigger_id, 1)

    assert {int(key) for (key, _) in training_samples_part0} == set(range(5))
    assert {int(key) for (key, _) in training_samples_part1} == set(range(5, 10))

    strat.inform_data(data2, timestamps2, labels2)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 20
    assert trigger_num_partitions == 4

    training_samples_part0 = strat.get_trigger_partition_keys(trigger_id, 0)
    training_samples_part1 = strat.get_trigger_partition_keys(trigger_id, 1)
    training_samples_part2 = strat.get_trigger_partition_keys(trigger_id, 2)
    training_samples_part3 = strat.get_trigger_partition_keys(trigger_id, 3)

    assert {int(key) for (key, _) in training_samples_part0} == set(range(5))
    assert {int(key) for (key, _) in training_samples_part1} == set(range(5, 10))
    assert {int(key) for (key, _) in training_samples_part2} == set(range(10, 15))
    assert {int(key) for (key, _) in training_samples_part3} == set(range(15, 20))


def test_e2e_reset_nolimit():
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels1 = [0] * 10

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))
    labels2 = [0] * 10

    # RESET // NO LIMIT #
    conf = get_config()
    conf.tail_triggers = 0
    conf.maximum_keys_in_memory = 100
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 10
    assert trigger_num_partitions == 1

    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)
    assert {int(key) for (key, _) in training_samples} == set(range(10))

    strat.inform_data(data2, timestamps2, labels2)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 10
    assert trigger_num_partitions == 1

    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)

    assert {int(key) for (key, _) in training_samples} == set(range(10, 20))


def test_e2e_reset_limit():
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels1 = [0] * 10

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))
    labels2 = [0] * 10

    # RESET // LIMIT #
    conf = get_config()
    conf.limit = 5
    conf.tail_triggers = 0
    conf.maximum_keys_in_memory = 100

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 5
    assert trigger_num_partitions == 1

    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)
    assert {int(key) for (key, _) in training_samples} < set(range(10))

    strat.inform_data(data2, timestamps2, labels2)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 5
    assert trigger_num_partitions == 1

    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)

    assert {int(key) for (key, _) in training_samples} < set(range(10, 20))


def test_e2e_reset_limit_uar():
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels1 = [0] * 10

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))
    labels2 = [0] * 10
    # NO RESET // LIMIT (UAR) #
    conf = get_config()
    conf.limit = 5
    conf.limit_reset = "sampleUAR"
    conf.maximum_keys_in_memory = 100

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 5
    assert trigger_num_partitions == 1

    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)
    assert {int(key) for (key, _) in training_samples} < set(range(10))

    strat.inform_data(data2, timestamps2, labels2)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 5
    assert trigger_num_partitions == 1

    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)
    assert {int(key) for (key, _) in training_samples} < set(range(20))


def test_e2e_reset_limit_lastx():
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels1 = [0] * 10

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))
    labels2 = [0] * 10

    # NO RESET // LIMIT (lastX) #
    conf = get_config()
    conf.limit = 5
    conf.limit_reset = "lastX"
    conf.maximum_keys_in_memory = 100
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 5
    assert trigger_num_partitions == 1

    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)
    assert {int(key) for (key, _) in training_samples} == set(range(5, 10))

    strat.inform_data(data2, timestamps2, labels2)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 5
    assert trigger_num_partitions == 1

    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)
    assert {int(key) for (key, _) in training_samples} == set(range(15, 20))


def test_e2e_reset_limit_lastx_large():
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels1 = [0] * 10

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))
    labels2 = [0] * 10

    # NO RESET // LIMIT (lastX w/ large limit) #
    conf = get_config()
    conf.limit = 15
    conf.limit_reset = "lastX"
    conf.maximum_keys_in_memory = 100

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 10
    assert trigger_num_partitions == 1

    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)
    assert {int(key) for (key, _) in training_samples} == set(range(10))

    strat.inform_data(data2, timestamps2, labels2)
    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
    assert trigger_num_keys == 15
    assert trigger_num_partitions == 1

    training_samples = strat.get_trigger_partition_keys(trigger_id, 0)
    assert {int(key) for (key, _) in training_samples} == set(range(5, 20))


def test_inform_data():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(
            SelectorStateMetadata.sample_key,
            SelectorStateMetadata.timestamp,
            SelectorStateMetadata.label,
            SelectorStateMetadata.pipeline_id,
            SelectorStateMetadata.used,
        ).all()

        assert len(data) == 0

    strat = NewDataStrategy(get_config(), get_minimal_modyn_config(), 0)
    strat.inform_data([10, 11, 12], [0, 1, 2], [0, 0, 1])

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(
            SelectorStateMetadata.sample_key,
            SelectorStateMetadata.timestamp,
            SelectorStateMetadata.label,
            SelectorStateMetadata.pipeline_id,
            SelectorStateMetadata.used,
        ).all()

        assert len(data) == 3

        keys, timestamps, labels, pipeline_ids, useds = zip(*data)

        assert not any(useds)
        for pip_id in pipeline_ids:
            assert pip_id == 0

        assert keys[0] == 10
        assert timestamps[0] == 0
        assert labels[0] == 0
        assert keys[1] == 11
        assert timestamps[1] == 1
        assert labels[1] == 0
        assert keys[2] == 12
        assert timestamps[2] == 2
        assert labels[2] == 1


@patch.object(NewDataStrategy, "_get_data_reset")
@patch.object(NewDataStrategy, "_get_data_no_reset")
def test__on_trigger_reset(test__get_data_no_reset: MagicMock, test__get_data_reset: MagicMock):
    conf = get_config()
    conf.tail_triggers = 0
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    assert strat.reset_after_trigger

    test__get_data_reset.return_value = [([10, 11, 12, 13], {})]
    test__get_data_no_reset.return_value = []

    result = list(strat._on_trigger())[0][0]

    test__get_data_no_reset.assert_not_called()
    test__get_data_reset.assert_called_once()

    result_keys = [key for (key, _) in result]
    result_weights = [weight for (_, weight) in result]

    assert sorted(result_keys) == [10, 11, 12, 13]
    for weight in result_weights:
        assert isclose(weight, 1.0)


@patch.object(NewDataStrategy, "_get_data_reset")
@patch.object(NewDataStrategy, "_get_data_no_reset")
def test__on_trigger_no_reset(test__get_data_no_reset: MagicMock, test__get_data_reset: MagicMock):
    conf = get_config()
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    assert not strat.reset_after_trigger

    test__get_data_reset.return_value = []
    test__get_data_no_reset.return_value = [([10, 11, 12, 13], {})]

    result = list(strat._on_trigger())[0][0]

    test__get_data_reset.assert_not_called()
    test__get_data_no_reset.assert_called_once()

    result_keys = [key for (key, _) in result]
    result_weights = [weight for (_, weight) in result]

    assert sorted(result_keys) == [10, 11, 12, 13]
    for weight in result_weights:
        assert isclose(weight, 1.0)


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
def test__get_data_reset_nolimit(test__get_all_data: MagicMock, test__get_current_trigger_data: MagicMock):
    conf = get_config()
    conf.tail_triggers = 0

    test__get_current_trigger_data.return_value = [([10, 11, 12], {})]
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert list(strat._get_data_reset()) == [([10, 11, 12], {})]
    test__get_all_data.assert_not_called()
    test__get_current_trigger_data.assert_called_once()


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
def test__get_data_reset_nolimit_partitions(test__get_all_data: MagicMock, test__get_current_trigger_data: MagicMock):
    conf = get_config()
    conf.tail_triggers = 0

    test__get_current_trigger_data.return_value = [
        ([10, 11, 12], {}),
        ([13, 14, 15], {}),
    ]
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert list(strat._get_data_reset()) == [([10, 11, 12], {}), ([13, 14, 15], {})]
    test__get_all_data.assert_not_called()
    test__get_current_trigger_data.assert_called_once()


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
def test__get_data_reset_smalllimit(test__get_all_data: MagicMock, test__get_current_trigger_data: MagicMock):
    conf = get_config()
    conf.limit = 2
    conf.tail_triggers = 0
    test__get_current_trigger_data.return_value = [([10, 11, 12], {})]

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    keys = list(strat._get_data_reset())[0][0]
    assert len(keys) == 2
    assert set(keys) < set([10, 11, 12])
    test__get_all_data.assert_not_called()
    test__get_current_trigger_data.assert_called_once()


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
def test__get_data_reset_largelimit(test__get_all_data: MagicMock, test__get_current_trigger_data: MagicMock):
    conf = get_config()
    conf.limit = 42
    conf.tail_triggers = 0
    test__get_current_trigger_data.return_value = [([10, 11, 12], {})]

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    keys = list(strat._get_data_reset())[0][0]
    assert len(keys) == 3
    assert set(keys) == set([10, 11, 12])
    test__get_all_data.assert_not_called()
    test__get_current_trigger_data.assert_called_once()


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
@patch.object(NewDataStrategy, "_handle_limit_no_reset")
def test__get_data_no_reset_nolimit(
    test__handle_limit_no_reset: MagicMock,
    test__get_all_data: MagicMock,
    test__get_current_trigger_data: MagicMock,
):
    conf = get_config()

    test__get_all_data.return_value = [([10, 11, 12], {})]
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert list(strat._get_data_no_reset())[0][0] == [10, 11, 12]
    test__get_all_data.assert_called_once()
    test__get_current_trigger_data.assert_not_called()
    test__handle_limit_no_reset.assert_not_called()


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
@patch.object(NewDataStrategy, "_handle_limit_no_reset")
def test__get_data_no_reset_limit(
    test__handle_limit_no_reset: MagicMock,
    test__get_all_data: MagicMock,
    test__get_current_trigger_data: MagicMock,
):
    conf = get_config()
    conf.limit = 2
    conf.limit_reset = "sampleUAR"

    test__handle_limit_no_reset.return_value = [10, 11]
    test__get_all_data.return_value = [([10, 11, 12], {})]
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert list(strat._get_data_no_reset())[0][0] == [10, 11]
    test__get_all_data.assert_called_once()
    test__get_current_trigger_data.assert_not_called()
    test__handle_limit_no_reset.assert_called_once()


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
@patch.object(NewDataStrategy, "_handle_limit_no_reset")
def test__get_data_no_reset_limit_partitions(
    test__handle_limit_no_reset: MagicMock,
    test__get_all_data: MagicMock,
    test__get_current_trigger_data: MagicMock,
):
    conf = get_config()
    conf.limit = 2
    conf.limit_reset = "sampleUAR"

    test__handle_limit_no_reset.side_effect = lambda x: [x[0], x[1]]
    test__get_all_data.return_value = [([10, 11, 12], {}), ([13, 14, 15], {})]
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert [data for data, _ in strat._get_data_no_reset()] == [[10, 11], [13, 14]]
    test__get_all_data.assert_called_once()
    test__get_current_trigger_data.assert_not_called()
    assert test__handle_limit_no_reset.call_count == 2


@patch.object(NewDataStrategy, "_last_x_limit")
@patch.object(NewDataStrategy, "_sample_uar")
def test__handle_limit_no_reset_lastx(test__sample_uar: MagicMock, test__last_x_limit: MagicMock):
    conf = get_config()
    conf.limit = 42
    conf.limit_reset = "lastX"
    test__last_x_limit.return_value = [10, 11]
    test__sample_uar.return_value = [12, 13]

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert strat._handle_limit_no_reset(["x"]) == [10, 11]
    test__last_x_limit.assert_called_once_with(["x"])
    test__sample_uar.assert_not_called()


@patch.object(NewDataStrategy, "_last_x_limit")
@patch.object(NewDataStrategy, "_sample_uar")
def test__handle_limit_no_reset_sampleuar(test__sample_uar: MagicMock, test__last_x_limit: MagicMock):
    conf = get_config()
    conf.limit = 42
    conf.limit_reset = "sampleUAR"
    test__last_x_limit.return_value = [10, 11]
    test__sample_uar.return_value = [12, 13]

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert strat._handle_limit_no_reset(["x"]) == [12, 13]
    test__last_x_limit.assert_not_called()
    test__sample_uar.assert_called_once_with(["x"])


def test__last_x_limit():
    conf = get_config()
    conf.limit = 5
    conf.limit_reset = "lastX"

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    assert strat._last_x_limit(list(range(1, 10))) == list(range(5, 10))


def test__sample_uar():
    conf = get_config()
    conf.limit = 5
    conf.limit_reset = "sampleUAR"

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    samples = strat._sample_uar(list(range(1, 10)))

    assert len(samples) == 5
    assert set(samples) < set(range(1, 10))


def test__get_current_trigger_data_no_partitions():
    strat = NewDataStrategy(get_config(), get_minimal_modyn_config(), 0)
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels = [0] * 10

    strat.inform_data(data1, timestamps1, labels)

    current_data = list(strat._get_current_trigger_data())[0][0]

    assert set(current_data) == set(data1)
    strat.trigger()

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))

    strat.inform_data(data2, timestamps2, labels)
    current_data = list(strat._get_current_trigger_data())[0][0]

    assert set(current_data) == set(data2)


def test__get_current_trigger_data_partitions():
    conf = get_config()
    conf.maximum_keys_in_memory = 1
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels = [0] * 10

    strat.inform_data(data1, timestamps1, labels)

    current_data = [data for data, _ in strat._get_current_trigger_data()]
    assert len(current_data) == 10
    current_data = flatten(current_data)

    assert set(current_data) == set(data1)
    strat.trigger()

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))

    strat.inform_data(data2, timestamps2, labels)
    current_data = [data for data, _ in strat._get_current_trigger_data()]
    assert len(current_data) == 10
    current_data = flatten(current_data)

    assert set(current_data) == set(data2)


def test__get_tail_triggers_data():
    conf = get_config_tail()
    conf.maximum_keys_in_memory = 1
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels = [0] * 10

    strat.inform_data(data1, timestamps1, labels)

    current_data = [data for data, _ in strat._get_tail_triggers_data()]
    assert len(current_data) == 10
    current_data = flatten(current_data)

    assert set(current_data) == set(data1)

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))

    strat.trigger()
    strat.inform_data(data2, timestamps2, labels)

    current_data = [data for data, _ in strat._get_tail_triggers_data()]
    assert len(current_data) == 20
    current_data = flatten(current_data)

    assert set(current_data) == set(data1 + data2)

    data3 = list(range(20, 30))
    timestamps3 = list(range(20, 30))

    strat.trigger()
    strat.inform_data(data3, timestamps3, labels)

    # since tail_trigger = 1 we should not get any point belonging to the first trigger
    current_data = [data for data, _ in strat._get_tail_triggers_data()]
    assert len(current_data) == 20
    current_data = flatten(current_data)

    assert set(current_data) == set(data2 + data3)

    data4 = list(range(30, 40))
    timestamps4 = list(range(30, 40))

    strat.trigger()
    strat.inform_data(data4, timestamps4, labels)

    # since tail_trigger = 1 we should not get any point belonging to the first and second trigger
    current_data = [data for data, _ in strat._get_tail_triggers_data()]
    assert len(current_data) == 20
    current_data = flatten(current_data)

    assert set(current_data) == set(data3 + data4)


def test__get_all_data_no_partitions():
    strat = NewDataStrategy(get_config(), get_minimal_modyn_config(), 0)
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels = [0] * 10

    strat.inform_data(data1, timestamps1, labels)
    all_data = list(strat._get_all_data())[0][0]

    assert set(all_data) == set(data1)
    strat.trigger()

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))

    strat.inform_data(data2, timestamps2, labels)
    all_data = list(strat._get_all_data())[0][0]
    assert all_data == data1 + data2


def test__get_all_data_partitions():
    conf = get_config()
    conf.maximum_keys_in_memory = 1
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels = [0] * 10

    strat.inform_data(data1, timestamps1, labels)
    all_data = [data for data, _ in strat._get_all_data()]
    assert len(all_data) == 10
    assert flatten(all_data) == data1

    strat.trigger()

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))

    strat.inform_data(data2, timestamps2, labels)
    all_data = [data for data, _ in strat._get_all_data()]
    assert len(all_data) == 20
    assert flatten(all_data) == data1 + data2


def test__get_all_data_partitions_with_same_timestamp():
    conf = get_config()
    conf.maximum_keys_in_memory = 1
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    data1 = list(range(10))
    timestamps1 = [42 for _ in range(10)]
    labels = [0] * 10

    strat.inform_data(data1, timestamps1, labels)
    all_data = [data for data, _ in strat._get_all_data()]
    assert len(all_data) == 10
    assert set(flatten(all_data)) == set(data1)
    assert len(flatten(all_data)) == len(data1)

    strat.trigger()

    data2 = list(range(10, 20))
    timestamps2 = [21 for _ in range(10)]

    strat.inform_data(data2, timestamps2, labels)
    all_data = [data for data, _ in strat._get_all_data()]
    assert len(all_data) == 20
    assert set(flatten(all_data)) == set(data1 + data2)
    assert len(flatten(all_data)) == len(data1 + data2)
