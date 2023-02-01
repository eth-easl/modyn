# pylint: disable=no-value-for-parameter,redefined-outer-name,singleton-comparison
import os
import pathlib
from math import isclose
from unittest.mock import MagicMock, patch

import pytest
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.metadata_database.models.training import Training
from modyn.backend.selector.internal.selector_strategies.new_data_strategy import NewDataStrategy

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


def get_config():
    return {"reset_after_trigger": False, "limit": -1}


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        training = Training(number_of_workers=1)
        database.session.add(training)
        database.session.commit()

    yield

    os.remove(database_path)


def test_constructor():
    NewDataStrategy(get_config(), get_minimal_modyn_config(), 0)


def test_constructor_throws_on_invalid_config():
    conf = get_config()
    conf["limit"] = 500

    with pytest.raises(ValueError):
        NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    conf = get_config()
    conf["limit"] = 500
    conf["limit_reset"] = "unknown"

    with pytest.raises(ValueError):
        NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    conf["limit_reset"] = "lastX"
    NewDataStrategy(conf, get_minimal_modyn_config(), 0)  # should work


def test_e2e_noreset_nolimit():
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels1 = [0] * 10

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))
    labels2 = [0] * 10

    # NO RESET // NO LIMIT #
    conf = get_config()
    conf["limit"] = -1
    conf["reset_after_trigger"] = False
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 10
    assert {int(key) for (key, _) in training_samples} == set(range(10))
    strat.inform_data(data2, timestamps2, labels2)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 20
    assert {int(key) for (key, _) in training_samples} == set(range(20))


def test_e2e_reset_nolimit():
    data1 = list(range(10))
    timestamps1 = list(range(10))
    labels1 = [0] * 10

    data2 = list(range(10, 20))
    timestamps2 = list(range(10, 20))
    labels2 = [0] * 10

    # RESET // NO LIMIT #
    conf = get_config()
    conf["limit"] = -1
    conf["reset_after_trigger"] = True
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 10
    assert {int(key) for (key, _) in training_samples} == set(range(10))
    strat.inform_data(data2, timestamps2, labels2)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 10
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
    conf["limit"] = 5
    conf["reset_after_trigger"] = True
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 5
    assert {int(key) for (key, _) in training_samples} < set(range(10))
    strat.inform_data(data2, timestamps2, labels2)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 5
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
    conf["limit"] = 5
    conf["reset_after_trigger"] = False
    conf["limit_reset"] = "sampleUAR"
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 5
    assert {int(key) for (key, _) in training_samples} < set(range(10))
    strat.inform_data(data2, timestamps2, labels2)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 5
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
    conf["limit"] = 5
    conf["reset_after_trigger"] = False
    conf["limit_reset"] = "lastX"
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 5
    assert {int(key) for (key, _) in training_samples} == set(range(5, 10))
    strat.inform_data(data2, timestamps2, labels2)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 5
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
    conf["limit"] = 15
    conf["reset_after_trigger"] = False
    conf["limit_reset"] = "lastX"
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    strat.inform_data(data1, timestamps1, labels1)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 10
    assert {int(key) for (key, _) in training_samples} == set(range(10))
    strat.inform_data(data2, timestamps2, labels2)
    _, training_samples = strat.trigger()
    assert len(training_samples) == 15
    assert {int(key) for (key, _) in training_samples} == set(range(5, 20))


def test_inform_data():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(
            Metadata.key, Metadata.timestamp, Metadata.label, Metadata.pipeline_id, Metadata.seen
        ).all()

        assert len(data) == 0

    strat = NewDataStrategy(get_config(), get_minimal_modyn_config(), 0)
    strat.inform_data(["a", "b", "c"], [0, 1, 2], [0, 0, 1])

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(
            Metadata.key, Metadata.timestamp, Metadata.label, Metadata.pipeline_id, Metadata.seen
        ).all()

        assert len(data) == 3

        keys, timestamps, labels, pipeline_ids, useds = zip(*data)

        assert not any(useds)
        for pip_id in pipeline_ids:
            assert pip_id == 0

        assert keys[0] == "a"
        assert timestamps[0] == 0
        assert labels[0] == 0
        assert keys[1] == "b"
        assert timestamps[1] == 1
        assert labels[1] == 0
        assert keys[2] == "c"
        assert timestamps[2] == 2
        assert labels[2] == 1


@patch.object(NewDataStrategy, "_get_data_reset")
@patch.object(NewDataStrategy, "_get_data_no_reset")
def test__on_trigger_reset(test__get_data_no_reset: MagicMock, test__get_data_reset: MagicMock):
    conf = get_config()
    conf["reset_after_trigger"] = True
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    assert strat.reset_after_trigger

    test__get_data_reset.return_value = ["a", "b", "c", "d"]
    test__get_data_no_reset.return_value = []

    result = strat._on_trigger()

    test__get_data_no_reset.assert_not_called()
    test__get_data_reset.assert_called_once()

    result_keys = [key for (key, _) in result]
    result_weights = [weight for (_, weight) in result]

    assert sorted(result_keys) == ["a", "b", "c", "d"]
    for weight in result_weights:
        assert isclose(weight, 1.0)


@patch.object(NewDataStrategy, "_get_data_reset")
@patch.object(NewDataStrategy, "_get_data_no_reset")
def test__on_trigger_no_reset(test__get_data_no_reset: MagicMock, test__get_data_reset: MagicMock):
    conf = get_config()
    conf["reset_after_trigger"] = False
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    assert not strat.reset_after_trigger

    test__get_data_reset.return_value = []
    test__get_data_no_reset.return_value = ["a", "b", "c", "d"]

    result = strat._on_trigger()

    test__get_data_reset.assert_not_called()
    test__get_data_no_reset.assert_called_once()

    result_keys = [key for (key, _) in result]
    result_weights = [weight for (_, weight) in result]

    assert sorted(result_keys) == ["a", "b", "c", "d"]
    for weight in result_weights:
        assert isclose(weight, 1.0)


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
def test__get_data_reset_nolimit(test__get_all_data: MagicMock, test__get_current_trigger_data: MagicMock):
    conf = get_config()
    conf["reset_after_trigger"] = True

    test__get_current_trigger_data.return_value = ["a", "b", "c"]
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert strat._get_data_reset() == ["a", "b", "c"]
    test__get_all_data.assert_not_called()
    test__get_current_trigger_data.assert_called_once()


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
def test__get_data_reset_smalllimit(test__get_all_data: MagicMock, test__get_current_trigger_data: MagicMock):
    conf = get_config()
    conf["reset_after_trigger"] = True
    conf["limit"] = 2
    test__get_current_trigger_data.return_value = ["a", "b", "c"]

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    keys = strat._get_data_reset()
    assert len(keys) == 2
    assert set(keys) < set(["a", "b", "c"])
    test__get_all_data.assert_not_called()
    test__get_current_trigger_data.assert_called_once()


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
def test__get_data_reset_largelimit(test__get_all_data: MagicMock, test__get_current_trigger_data: MagicMock):
    conf = get_config()
    conf["reset_after_trigger"] = True
    conf["limit"] = 42
    test__get_current_trigger_data.return_value = ["a", "b", "c"]

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    keys = strat._get_data_reset()
    assert len(keys) == 3
    assert set(keys) == set(["a", "b", "c"])
    test__get_all_data.assert_not_called()
    test__get_current_trigger_data.assert_called_once()


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
@patch.object(NewDataStrategy, "_handle_limit_no_reset")
def test__get_data_no_reset_nolimit(
    test__handle_limit_no_reset: MagicMock, test__get_all_data: MagicMock, test__get_current_trigger_data: MagicMock
):
    conf = get_config()
    conf["reset_after_trigger"] = False

    test__get_all_data.return_value = ["a", "b", "c"]
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert strat._get_data_no_reset() == ["a", "b", "c"]
    test__get_all_data.assert_called_once()
    test__get_current_trigger_data.assert_not_called()
    test__handle_limit_no_reset.assert_not_called()


@patch.object(NewDataStrategy, "_get_current_trigger_data")
@patch.object(NewDataStrategy, "_get_all_data")
@patch.object(NewDataStrategy, "_handle_limit_no_reset")
def test__get_data_no_reset_limit(
    test__handle_limit_no_reset: MagicMock, test__get_all_data: MagicMock, test__get_current_trigger_data: MagicMock
):
    conf = get_config()
    conf["reset_after_trigger"] = False
    conf["limit"] = 2
    conf["limit_reset"] = "sampleUAR"

    test__handle_limit_no_reset.return_value = ["a", "b"]
    test__get_all_data.return_value = ["a", "b", "c"]
    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert strat._get_data_no_reset() == ["a", "b"]
    test__get_all_data.assert_called_once()
    test__get_current_trigger_data.assert_not_called()
    test__handle_limit_no_reset.assert_called_once()


@patch.object(NewDataStrategy, "_last_x_limit")
@patch.object(NewDataStrategy, "_sample_uar")
def test__handle_limit_no_reset_lastx(test__sample_uar: MagicMock, test__last_x_limit: MagicMock):
    conf = get_config()
    conf["reset_after_trigger"] = False
    conf["limit"] = 42
    conf["limit_reset"] = "lastX"
    test__last_x_limit.return_value = ["a", "b"]
    test__sample_uar.return_value = ["c", "d"]

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert strat._handle_limit_no_reset(["x"]) == ["a", "b"]
    test__last_x_limit.assert_called_once_with(["x"])
    test__sample_uar.assert_not_called()


@patch.object(NewDataStrategy, "_last_x_limit")
@patch.object(NewDataStrategy, "_sample_uar")
def test__handle_limit_no_reset_sampleuar(test__sample_uar: MagicMock, test__last_x_limit: MagicMock):
    conf = get_config()
    conf["reset_after_trigger"] = False
    conf["limit"] = 42
    conf["limit_reset"] = "sampleUAR"
    test__last_x_limit.return_value = ["a", "b"]
    test__sample_uar.return_value = ["c", "d"]

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)

    assert strat._handle_limit_no_reset(["x"]) == ["c", "d"]
    test__last_x_limit.assert_not_called()
    test__sample_uar.assert_called_once_with(["x"])


def test__last_x_limit():
    conf = get_config()
    conf["limit"] = 5
    conf["limit_reset"] = "lastX"

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    assert strat._last_x_limit(list(range(1, 10))) == list(range(5, 10))


def test__sample_uar():
    conf = get_config()
    conf["limit"] = 5
    conf["limit_reset"] = "sampleUAR"

    strat = NewDataStrategy(conf, get_minimal_modyn_config(), 0)
    samples = strat._sample_uar(list(range(1, 10)))

    assert len(samples) == 5
    assert set(samples) < set(range(1, 10))


def test__get_current_trigger_data():
    strat = NewDataStrategy(get_config(), get_minimal_modyn_config(), 0)
    data1 = [str(x) for x in list(range(10))]
    timestamps1 = list(range(10))
    labels = [0] * 10

    strat.inform_data(data1, timestamps1, labels)

    assert set(strat._get_current_trigger_data()) == set(data1)
    strat.trigger()

    data2 = [str(x) for x in list(range(10, 20))]
    timestamps2 = list(range(10, 20))

    strat.inform_data(data2, timestamps2, labels)
    assert set(strat._get_current_trigger_data()) == set(data2)


def test__get_all_data():
    strat = NewDataStrategy(get_config(), get_minimal_modyn_config(), 0)
    data1 = [str(x) for x in list(range(10))]
    timestamps1 = list(range(10))
    labels = [0] * 10

    strat.inform_data(data1, timestamps1, labels)

    assert set(strat._get_all_data()) == set(data1)
    strat.trigger()

    data2 = [str(x) for x in list(range(10, 20))]
    timestamps2 = list(range(10, 20))

    strat.inform_data(data2, timestamps2, labels)
    assert set(strat._get_all_data()) == set(data1 + data2)


def test__reset_state():
    strat = NewDataStrategy(get_config(), get_minimal_modyn_config(), 0)
    strat._reset_state()
