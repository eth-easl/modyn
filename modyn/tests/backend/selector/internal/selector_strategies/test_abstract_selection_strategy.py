# pylint: disable=no-value-for-parameter,redefined-outer-name,abstract-class-instantiated
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SelectorStateMetadata, Trigger, TriggerSample
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy

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


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield

    os.remove(database_path)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test_init():
    # Test init works
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 42)
    assert not strat.has_limit
    assert not strat.reset_after_trigger
    assert strat._pipeline_id == 42
    assert strat._next_trigger_id == 0

    # Test required config check works
    with pytest.raises(ValueError):
        AbstractSelectionStrategy({"limit": -1}, get_minimal_modyn_config(), 42)

    with pytest.raises(ValueError):
        AbstractSelectionStrategy(
            {"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 42, ["doesntexist"]
        )

    #  Test reinit works
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 42)
    strat._next_trigger_id = 1


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test__on_trigger():
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 42)
    with pytest.raises(NotImplementedError):
        strat._on_trigger()


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test__reset_state():
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 42)
    with pytest.raises(NotImplementedError):
        strat._reset_state()


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test_inform_data():
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 42)
    with pytest.raises(NotImplementedError):
        strat.inform_data([], [], [])


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_without_reset(test_reset_state: MagicMock, test__on_trigger: MagicMock):
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 42)
    assert not strat.reset_after_trigger
    assert strat._next_trigger_id == 0

    test__on_trigger.return_value = [[("a", 1.0), ("b", 1.0), ("c", 1.0)]]

    trigger_id, trigger_num_keys, trigger_num_partitions = strat.trigger()

    assert trigger_id == 0
    assert strat._next_trigger_id == 1
    assert trigger_num_keys == 3
    assert trigger_num_partitions == 1

    test_reset_state.assert_not_called()
    test__on_trigger.assert_called_once()

    assert strat.get_trigger_partition_keys(trigger_id, 0) == [("a", 1.0), ("b", 1.0), ("c", 1.0)]


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_without_reset_multiple_partitions(test_reset_state: MagicMock, test__on_trigger: MagicMock):
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 42)
    assert not strat.reset_after_trigger
    assert strat._next_trigger_id == 0

    test__on_trigger.return_value = [[("a", 1.0), ("b", 1.0), ("c", 1.0)], [("d", 1.0), ("e", 1.0), ("f", 1.0)]]

    trigger_id, trigger_num_keys, trigger_num_partitions = strat.trigger()

    assert trigger_id == 0
    assert strat._next_trigger_id == 1
    assert trigger_num_keys == 6
    assert trigger_num_partitions == 2

    test_reset_state.assert_not_called()
    test__on_trigger.assert_called_once()

    assert strat.get_trigger_partition_keys(trigger_id, 0) == [("a", 1.0), ("b", 1.0), ("c", 1.0)]
    assert strat.get_trigger_partition_keys(trigger_id, 1) == [("d", 1.0), ("e", 1.0), ("f", 1.0)]
    assert strat.get_trigger_partition_keys(trigger_id, 2) == []


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_with_reset(test_reset_state: MagicMock, test__on_trigger: MagicMock):
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": True}, get_minimal_modyn_config(), 42)
    assert strat.reset_after_trigger
    assert strat._next_trigger_id == 0

    test__on_trigger.return_value = [[("a", 1.0), ("b", 1.0), ("c", 1.0)]]

    trigger_id, trigger_num_keys, trigger_num_partitions = strat.trigger()

    assert trigger_id == 0
    assert trigger_id == 0
    assert strat._next_trigger_id == 1
    assert trigger_num_keys == 3
    assert trigger_num_partitions == 1

    test_reset_state.assert_called_once()
    test__on_trigger.assert_called_once()
    assert strat.get_trigger_partition_keys(trigger_id, 0) == [("a", 1.0), ("b", 1.0), ("c", 1.0)]


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_trigger_stored(_: MagicMock, test__on_trigger: MagicMock):
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": True}, get_minimal_modyn_config(), 42)
    assert strat.reset_after_trigger
    assert strat._next_trigger_id == 0

    test__on_trigger.return_value = [[("a", 1.0), ("b", 1.0), ("c", 1.0)], [("d", 1.0)]]

    trigger_id, trigger_num_keys, trigger_num_partitions = strat.trigger()
    assert trigger_id == 0
    assert trigger_num_keys == 4
    assert trigger_num_partitions == 2
    assert strat._next_trigger_id == 1

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(Trigger).all()

        assert len(data) == 1
        assert data[0].trigger_id == 0
        assert data[0].pipeline_id == 42
        assert data[0].num_keys == 4
        assert data[0].num_partitions == 2

        data = database.session.query(TriggerSample).all()

        assert len(data) == 4
        assert data[0].trigger_id == 0
        assert data[0].sample_key == "a"
        assert data[0].pipeline_id == 42
        assert data[0].partition_id == 0

        assert data[1].trigger_id == 0
        assert data[1].sample_key == "b"
        assert data[1].pipeline_id == 42
        assert data[1].partition_id == 0

        assert data[2].trigger_id == 0
        assert data[2].sample_key == "c"
        assert data[2].pipeline_id == 42
        assert data[2].partition_id == 0

        assert data[3].trigger_id == 0
        assert data[3].sample_key == "d"
        assert data[3].pipeline_id == 42
        assert data[3].partition_id == 1


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test__persist_data():
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": True}, get_minimal_modyn_config(), 42)
    strat._persist_samples(["a", "b", "c"], [0, 1, 2], ["dog", "dog", "cat"])

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
            assert pip_id == 42

        assert keys[0] == "a" and keys[1] == "b" and keys[2] == "c"
        assert timestamps[0] == 0 and timestamps[1] == 1 and timestamps[2] == 2
        assert labels[0] == "dog" and labels[1] == "dog" and labels[2] == "cat"


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
def test_two_strategies_increase_next_trigger_separately(test__on_trigger: MagicMock):
    test__on_trigger.return_value = []

    strat1 = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 42)
    assert strat1._pipeline_id == 42
    assert strat1._next_trigger_id == 0

    strat1.trigger()
    assert strat1._next_trigger_id == 1
    strat1.trigger()
    assert strat1._next_trigger_id == 2

    strat2 = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 21)
    assert strat2._pipeline_id == 21
    assert strat2._next_trigger_id == 0

    strat2.trigger()
    assert strat2._next_trigger_id == 1
    assert strat1._next_trigger_id == 2
    strat1.trigger()
    assert strat1._next_trigger_id == 3
    assert strat2._next_trigger_id == 1
    strat2.trigger()
    assert strat1._next_trigger_id == 3
    assert strat2._next_trigger_id == 2
