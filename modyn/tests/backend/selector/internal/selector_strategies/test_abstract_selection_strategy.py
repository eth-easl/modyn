# pylint: disable=no-value-for-parameter,redefined-outer-name,abstract-class-instantiated
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SelectorStateMetadata
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

    test__on_trigger.return_value = [("a", 1.0), ("b", 1.0), ("c", 1.0)]

    trigger_id, samples = strat.trigger()

    assert trigger_id == 0
    assert strat._next_trigger_id == 1
    test_reset_state.assert_not_called()
    test__on_trigger.assert_called_once()
    assert samples == [("a", 1.0), ("b", 1.0), ("c", 1.0)]


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_with_reset(test_reset_state: MagicMock, test__on_trigger: MagicMock):
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": True}, get_minimal_modyn_config(), 42)
    assert strat.reset_after_trigger
    assert strat._next_trigger_id == 0

    test__on_trigger.return_value = [("a", 1.0), ("b", 1.0), ("c", 1.0)]

    trigger_id, samples = strat.trigger()

    assert trigger_id == 0
    assert strat._next_trigger_id == 1
    test_reset_state.assert_called_once()
    test__on_trigger.assert_called_once()
    assert samples == [("a", 1.0), ("b", 1.0), ("c", 1.0)]


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test__persist_data():
    strat = AbstractSelectionStrategy({"limit": -1, "reset_after_trigger": True}, get_minimal_modyn_config(), 42)
    strat._persist_data(["a", "b", "c"], [0, 1, 2], ["dog", "dog", "cat"])

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
