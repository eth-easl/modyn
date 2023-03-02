# pylint: disable=no-value-for-parameter,redefined-outer-name,singleton-comparison
import os
import pathlib
import random
from math import isclose
from unittest.mock import MagicMock, patch

import pytest
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SelectorStateMetadata
from modyn.backend.selector.internal.selector_strategies.freshness_sampling_strategy import FreshnessSamplingStrategy
from modyn.utils import flatten

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"


def get_minimal_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "host": "",
            "port": "0",
            "database": f"{database_path}",
        },
    }


def get_freshness_config():
    return {"reset_after_trigger": False, "unused_data_ratio": 50, "limit": -1, "maximum_keys_in_memory": 1000}


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield
    os.remove(database_path)


def test_constructor():
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    assert strat._is_first_trigger


def test_constructor_throws_on_invalid_config():
    conf = get_freshness_config()
    conf["reset_after_trigger"] = True

    with pytest.raises(ValueError):
        FreshnessSamplingStrategy(conf, get_minimal_modyn_config(), 0)

    conf = get_freshness_config()
    conf["unused_data_ratio"] = 0

    with pytest.raises(ValueError):
        FreshnessSamplingStrategy(conf, get_minimal_modyn_config(), 0)


def test_inform_data():
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    strat.inform_data(["a", "b", "c"], [0, 1, 2], ["dog", "dog", "cat"])

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

        assert keys[0] == "a" and keys[1] == "b" and keys[2] == "c"
        assert timestamps[0] == 0 and timestamps[1] == 1 and timestamps[2] == 2
        assert labels[0] == "dog" and labels[1] == "dog" and labels[2] == "cat"


@patch.object(FreshnessSamplingStrategy, "_get_first_trigger_data")
@patch.object(FreshnessSamplingStrategy, "_get_trigger_data")
@patch.object(FreshnessSamplingStrategy, "_mark_used")
def test__on_trigger_first_trigger(
    test__mark_used: MagicMock, test__get_trigger_data: MagicMock, test__get_first_trigger_data: MagicMock
):
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    strat._is_first_trigger = True

    test__mark_used.return_value = None
    test__get_trigger_data.return_value = []
    test__get_first_trigger_data.return_value = [["a", "b", "c", "d"]]

    result = list(strat._on_trigger())
    assert len(result) == 1
    result = result[0]

    test__get_trigger_data.assert_not_called()
    test__mark_used.assert_called_once()

    result_keys = [key for (key, _) in result]
    result_weights = [weight for (_, weight) in result]

    assert sorted(result_keys) == ["a", "b", "c", "d"]
    for weight in result_weights:
        assert isclose(weight, 1.0)


@patch.object(FreshnessSamplingStrategy, "_get_first_trigger_data")
@patch.object(FreshnessSamplingStrategy, "_get_trigger_data")
@patch.object(FreshnessSamplingStrategy, "_mark_used")
def test__on_trigger_subsequent_trigger(
    test__mark_used: MagicMock, test__get_trigger_data: MagicMock, test__get_first_trigger_data: MagicMock
):
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    strat._is_first_trigger = False

    test__mark_used.return_value = None
    test__get_trigger_data.return_value = [["a", "b", "c", "d"]]
    test__get_first_trigger_data.return_value = []

    result = list(strat._on_trigger())
    assert len(result) == 1
    result = result[0]

    test__get_first_trigger_data.assert_not_called()
    test__mark_used.assert_called_once()

    result_keys = [key for (key, _) in result]
    result_weights = [weight for (_, weight) in result]

    assert sorted(result_keys) == ["a", "b", "c", "d"]
    for weight in result_weights:
        assert isclose(weight, 1.0)


@patch.object(FreshnessSamplingStrategy, "_get_all_unused_data")
@patch.object(FreshnessSamplingStrategy, "_calc_num_samples_no_limit")
@patch.object(FreshnessSamplingStrategy, "_calc_num_samples_limit")
def test__get_first_trigger_data(
    test__calc_num_samples_limit: MagicMock,
    test__calc_num_samples_no_limit: MagicMock,
    test__get_all_unused_data: MagicMock,
):
    test__get_all_unused_data.return_value = [["a", "b", "c", "d"]]
    test__calc_num_samples_no_limit.return_value = (4, 4)
    test__calc_num_samples_limit.return_value = (2, 2)
    config = get_freshness_config()

    strat = FreshnessSamplingStrategy(config, get_minimal_modyn_config(), 0)
    strat._is_first_trigger = True

    result = list(strat._get_first_trigger_data())
    assert len(result) == 1
    result = result[0]

    test__get_all_unused_data.assert_called_once()

    assert not strat._is_first_trigger
    assert result == ["a", "b", "c", "d"]

    with pytest.raises(AssertionError):
        list(strat._get_first_trigger_data())

    strat.has_limit = True
    strat.training_set_size_limit = 20000
    strat._is_first_trigger = True
    result = list(strat._get_first_trigger_data())
    assert len(result) == 1
    result = result[0]
    assert result == ["a", "b", "c", "d"]

    strat.training_set_size_limit = 2
    strat._is_first_trigger = True
    result = list(strat._get_first_trigger_data())
    assert len(result) == 1
    result = result[0]

    assert len(result) == 2
    assert set(result) < set(["a", "b", "c", "d"])


@patch.object(FreshnessSamplingStrategy, "_get_all_unused_data")
@patch.object(FreshnessSamplingStrategy, "_calc_num_samples_no_limit")
@patch.object(FreshnessSamplingStrategy, "_calc_num_samples_limit")
def test__get_first_trigger_data_partitions(
    test__calc_num_samples_limit: MagicMock,
    test__calc_num_samples_no_limit: MagicMock,
    test__get_all_unused_data: MagicMock,
):
    test__get_all_unused_data.return_value = [["a", "b"], ["c", "d"]]
    test__calc_num_samples_no_limit.return_value = (4, 4)
    test__calc_num_samples_limit.return_value = (2, 2)
    config = get_freshness_config()

    strat = FreshnessSamplingStrategy(config, get_minimal_modyn_config(), 0)
    strat._is_first_trigger = True

    result = list(strat._get_first_trigger_data())
    assert len(result) == 2
    result = flatten(result)

    test__get_all_unused_data.assert_called_once()

    assert not strat._is_first_trigger
    assert result == ["a", "b", "c", "d"]

    with pytest.raises(AssertionError):
        list(strat._get_first_trigger_data())

    # TODO(create issue): We do not support a limit larger than or equal to the partition size right now


@patch.object(FreshnessSamplingStrategy, "_get_data_sample")
@patch.object(FreshnessSamplingStrategy, "_get_count_of_data")
@patch.object(FreshnessSamplingStrategy, "_calc_num_samples_no_limit")
@patch.object(FreshnessSamplingStrategy, "_calc_num_samples_limit")
def test__get_trigger_data_limit(
    test__calc_num_samples_limit: MagicMock,
    test__calc_num_samples_no_limit: MagicMock,
    test__get_count_of_data: MagicMock,
    test__get_data_sample: MagicMock,
):
    test__get_count_of_data.return_value = 4
    test__calc_num_samples_no_limit.return_value = (4, 4)
    test__calc_num_samples_limit.return_value = (2, 2)

    def sampler(size, used):
        if used:
            return iter([random.sample(["e", "f", "g", "h"], size)])

        return iter([random.sample(["a", "b", "c", "d"], size)])

    test__get_data_sample.side_effect = sampler

    config = get_freshness_config()
    config["limit"] = 4

    strat = FreshnessSamplingStrategy(config, get_minimal_modyn_config(), 0)
    strat._is_first_trigger = False
    result = list(strat._get_trigger_data())
    assert len(result) == 1
    result = result[0]

    assert len(result) == 4
    assert set(result) < set(["a", "b", "c", "d", "e", "f", "g", "h"])


@patch.object(FreshnessSamplingStrategy, "_get_data_sample")
@patch.object(FreshnessSamplingStrategy, "_get_count_of_data")
@patch.object(FreshnessSamplingStrategy, "_calc_num_samples_no_limit")
@patch.object(FreshnessSamplingStrategy, "_calc_num_samples_limit")
def test__get_trigger_data_no_limit(
    test__calc_num_samples_limit: MagicMock,
    test__calc_num_samples_no_limit: MagicMock,
    test__get_count_of_data: MagicMock,
    test__get_data_sample: MagicMock,
):
    test__get_count_of_data.return_value = 4
    test__calc_num_samples_no_limit.return_value = (4, 4)
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    strat._is_first_trigger = False

    def sampler(size, used):
        if used:
            return iter([random.sample(["e", "f", "g", "h"], size)])

        return iter([random.sample(["a", "b", "c", "d"], size)])

    test__get_data_sample.side_effect = sampler

    result = list(strat._get_trigger_data())
    assert len(result) == 1
    result = result[0]

    assert set(result) == set(["a", "b", "c", "d", "e", "f", "g", "h"])
    test__calc_num_samples_no_limit.assert_called_once_with(4, 4)
    test__calc_num_samples_limit.assert_not_called()

    test__calc_num_samples_no_limit.return_value = (2, 2)
    result = list(strat._get_trigger_data())
    assert len(result) == 1
    result = result[0]

    assert len(result) == 4
    assert set(result) < set(["a", "b", "c", "d", "e", "f", "g", "h"])

    with pytest.raises(AssertionError):
        strat._is_first_trigger = True
        list(strat._get_trigger_data())


def test__calc_num_samples_no_limit():
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    strat.unused_data_ratio = 50

    assert strat._calc_num_samples_no_limit(100, 100) == (100, 100)
    assert strat._calc_num_samples_no_limit(100, 200) == (100, 100)
    assert strat._calc_num_samples_no_limit(10, 200) == (10, 10)
    assert strat._calc_num_samples_no_limit(200, 100) == (100, 100)
    assert strat._calc_num_samples_no_limit(200, 10) == (10, 10)

    strat.unused_data_ratio = 90
    assert strat._calc_num_samples_no_limit(100, 100) == (100, 11)

    strat.unused_data_ratio = 10
    assert strat._calc_num_samples_no_limit(100, 100) == (11, 100)


def test__calc_num_samples_limit():
    config = get_freshness_config()
    config["limit"] = 100

    strat = FreshnessSamplingStrategy(config, get_minimal_modyn_config(), 0)
    strat.unused_data_ratio = 50

    assert strat.has_limit
    assert strat._calc_num_samples_limit(100, 100) == (50, 50)

    strat.training_set_size_limit = 50
    assert strat._calc_num_samples_limit(100, 100) == (25, 25)

    strat.training_set_size_limit = 100
    strat.unused_data_ratio = 90
    assert strat._calc_num_samples_limit(100, 100) == (90, 10)


def test__get_all_unused_data():
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    strat.inform_data(["a", "b", "c"], [0, 1, 2], ["dog", "dog", "cat"])
    strat._mark_used(["a"])

    assert set(list(strat._get_all_unused_data())[0]) == set(["b", "c"])
    strat._mark_used(["a", "b", "c"])
    assert len(list(strat._get_all_unused_data())) == 0


def test__get_all_unused_data_partitioning():
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    strat._maximum_keys_in_memory = 1
    strat.inform_data(["a", "b", "c"], [0, 1, 2], ["dog", "dog", "cat"])
    strat._mark_used(["a"])

    result = list(strat._get_all_unused_data())
    assert len(result) == 2
    result = flatten(result)

    assert set(result) == set(["b", "c"])
    strat._mark_used(["a", "b", "c"])
    assert len(list(strat._get_all_unused_data())) == 0


def test__get_data_sample_no_partitions():
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    strat.inform_data(["a", "b", "c", "d", "e", "f"], [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0])
    used = ["a", "b", "c"]
    unused = ["d", "e", "f"]

    strat._mark_used(used)

    # Confirm that mark used worked
    result = flatten(list(strat._get_all_unused_data()))
    assert set(result) == set(["d", "e", "f"])

    # Test full sample (not actually random)
    used_sample = list(strat._get_data_sample(3, True))
    assert len(used_sample) == 1
    used_sample = used_sample[0]
    assert len(used_sample) == 3
    assert set(used_sample) == set(used)

    unused_sample = list(strat._get_data_sample(3, False))
    assert len(unused_sample) == 1
    unused_sample = unused_sample[0]
    assert len(unused_sample) == 3
    assert set(unused_sample) == set(unused)

    # Test sample of size 2
    used_sample = list(strat._get_data_sample(2, True))
    assert len(used_sample) == 1
    used_sample = used_sample[0]
    assert len(used_sample) == 2
    assert set(used_sample) < set(used)

    unused_sample = list(strat._get_data_sample(2, False))
    assert len(unused_sample) == 1
    unused_sample = unused_sample[0]
    assert len(unused_sample) == 2
    assert set(unused_sample) < set(unused)


def test__get_data_sample_partitions():
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    strat._maximum_keys_in_memory = 1
    strat.inform_data(["a", "b", "c", "d", "e", "f"], [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0])
    used = ["a", "b", "c"]
    unused = ["d", "e", "f"]

    strat._mark_used(used)

    # Confirm that mark used worked
    result = flatten(list(strat._get_all_unused_data()))
    assert set(result) == set(["d", "e", "f"])

    # Test full sample (not actually random)
    used_sample = list(strat._get_data_sample(3, True))
    assert len(used_sample) == 3
    used_sample = flatten(used_sample)
    assert len(used_sample) == 3
    assert set(used_sample) == set(used)

    unused_sample = list(strat._get_data_sample(3, False))
    assert len(unused_sample) == 3
    unused_sample = flatten(unused_sample)
    assert len(unused_sample) == 3
    assert set(unused_sample) == set(unused)

    # Test sample of size 2
    used_sample = list(strat._get_data_sample(2, True))
    assert len(used_sample) == 2
    used_sample = flatten(used_sample)
    assert len(used_sample) == 2
    assert set(used_sample) < set(used)

    unused_sample = list(strat._get_data_sample(2, False))
    assert len(unused_sample) == 2
    unused_sample = flatten(unused_sample)
    assert len(unused_sample) == 2
    assert set(unused_sample) < set(unused)


def test__mark_used():
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    strat.inform_data(["a", "b", "c"], [0, 1, 2], ["dog", "dog", "cat"])

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(SelectorStateMetadata.used).all()

        assert len(data) == 3
        useds = [used[0] for used in data]
        assert not any(useds)

    strat._mark_used(["a", "c"])
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = (
            database.session.query(SelectorStateMetadata.sample_key, SelectorStateMetadata.used)
            .filter(SelectorStateMetadata.used == True)  # noqa: E712
            .all()
        )

        assert len(data) == 2
        keys, useds = zip(*data)

        assert all(useds)
        assert set(["a", "c"]) == set(keys)

        data = (
            database.session.query(SelectorStateMetadata.sample_key, SelectorStateMetadata.used)
            .filter(SelectorStateMetadata.used == False)  # noqa: E712
            .all()
        )

        assert len(data) == 1
        keys, useds = zip(*data)

        assert not any(useds)
        assert set(["b"]) == set(keys)


def test__reset_state():
    strat = FreshnessSamplingStrategy(get_freshness_config(), get_minimal_modyn_config(), 0)
    with pytest.raises(Exception):
        strat._reset_state()
