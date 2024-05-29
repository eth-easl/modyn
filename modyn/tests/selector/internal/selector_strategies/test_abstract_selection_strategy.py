# pylint: disable=no-value-for-parameter,redefined-outer-name,abstract-class-instantiated
import os
import pathlib
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from modyn.common.trigger_sample import TriggerSampleStorage
from modyn.config.schema.pipeline import _BaseSelectionStrategy
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Trigger, TriggerPartition
from modyn.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy

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


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield

    os.remove(database_path)
    shutil.rmtree(TMP_DIR)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test_init():
    # Test init works
    config = _BaseSelectionStrategy(
        limit=-1,
        tail_triggers=None,
        maximum_keys_in_memory=1000,
    )
    strat = AbstractSelectionStrategy(
        config,
        get_minimal_modyn_config(),
        42,
    )
    assert not strat.has_limit
    assert not strat.reset_after_trigger
    assert strat._pipeline_id == 42
    assert strat._next_trigger_id == 0
    assert strat.maximum_keys_in_memory == 1000

    # Test reinit works
    strat = AbstractSelectionStrategy(
        config,
        get_minimal_modyn_config(),
        42,
    )
    strat._next_trigger_id = 1


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_without_reset(test_reset_state: MagicMock, test__on_trigger: MagicMock):
    config = _BaseSelectionStrategy(
        limit=-1,
        tail_triggers=None,
        maximum_keys_in_memory=1000,
    )
    strat = AbstractSelectionStrategy(
        config,
        get_minimal_modyn_config(),
        42,
    )
    assert not strat.reset_after_trigger
    assert strat._next_trigger_id == 0

    test__on_trigger.return_value = [([(10, 1.0), (11, 1.0), (12, 1.0)], {})]

    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()

    assert trigger_id == 0
    assert strat._next_trigger_id == 1
    assert trigger_num_keys == 3
    assert trigger_num_partitions == 1

    test_reset_state.assert_not_called()
    test__on_trigger.assert_called_once()

    assert strat.get_trigger_partition_keys(trigger_id, 0) == np.array(
        [(10, 1.0), (11, 1.0), (12, 1.0)], dtype=[("f0", "<i8"), ("f1", "<f8")]
    )


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_without_reset_multiple_partitions(test_reset_state: MagicMock, test__on_trigger: MagicMock):
    config = _BaseSelectionStrategy(
        limit=-1,
        tail_triggers=None,
        maximum_keys_in_memory=1000,
    )
    strat = AbstractSelectionStrategy(
        config,
        get_minimal_modyn_config(),
        42,
    )
    assert not strat.reset_after_trigger
    assert strat._next_trigger_id == 0

    test__on_trigger.return_value = [
        ([(10, 1.0), (11, 1.0), (12, 1.0)], {}),
        ([(13, 1.0), (14, 1.0), (15, 1.0)], {}),
    ]

    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()

    assert trigger_id == 0
    assert strat._next_trigger_id == 1
    assert trigger_num_keys == 6
    assert trigger_num_partitions == 2

    test_reset_state.assert_not_called()
    test__on_trigger.assert_called_once()

    assert strat.get_trigger_partition_keys(trigger_id, 0) == np.array(
        [(10, 1.0), (11, 1.0), (12, 1.0)], dtype=[("f0", "<i8"), ("f1", "<f8")]
    )
    assert strat.get_trigger_partition_keys(trigger_id, 1) == np.array(
        [(13, 1.0), (14, 1.0), (15, 1.0)], dtype=[("f0", "<i8"), ("f1", "<f8")]
    )

    with pytest.raises(AssertionError):
        strat.get_trigger_partition_keys(trigger_id, 2)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_with_reset(test_reset_state: MagicMock, test__on_trigger: MagicMock):
    config = _BaseSelectionStrategy(
        limit=-1,
        tail_triggers=0,
        maximum_keys_in_memory=1000,
    )
    strat = AbstractSelectionStrategy(
        config,
        get_minimal_modyn_config(),
        42,
    )
    assert strat.reset_after_trigger
    assert strat._next_trigger_id == 0

    test__on_trigger.return_value = [([(10, 1.0), (11, 1.0), (12, 1.0)], {})]

    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()

    assert trigger_id == 0
    assert trigger_id == 0
    assert strat._next_trigger_id == 1
    assert trigger_num_keys == 3
    assert trigger_num_partitions == 1

    test_reset_state.assert_called_once()
    test__on_trigger.assert_called_once()
    assert strat.get_trigger_partition_keys(trigger_id, 0) == np.array(
        [(10, 1.0), (11, 1.0), (12, 1.0)], dtype=[("f0", "<i8"), ("f1", "<f8")]
    )


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_trigger_stored(_: MagicMock, test__on_trigger: MagicMock):
    config = _BaseSelectionStrategy(
        limit=-1,
        tail_triggers=0,
        maximum_keys_in_memory=1000,
    )
    strat = AbstractSelectionStrategy(config, get_minimal_modyn_config(), 42)
    assert strat.reset_after_trigger
    assert strat._next_trigger_id == 0

    test__on_trigger.return_value = [
        ([(10, 1.0), (11, 1.0), (12, 1.0)], {}),
        ([(13, 1.0)], {}),
    ]

    trigger_id, trigger_num_keys, trigger_num_partitions, _ = strat.trigger()
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

        data = TriggerSampleStorage(TMP_DIR).get_trigger_samples(42, 0, 0)

        assert len(data) == 3
        assert data[0][0] == 10
        assert data[0][1] == 1.0
        assert data[1][0] == 11
        assert data[1][1] == 1.0
        assert data[2][0] == 12
        assert data[2][1] == 1.0

        data = TriggerSampleStorage(TMP_DIR).get_trigger_samples(42, 0, 1)

        assert len(data) == 1
        assert data[0][0] == 13
        assert data[0][1] == 1.0


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
def test_two_strategies_increase_next_trigger_separately(test__on_trigger: MagicMock):
    test__on_trigger.return_value = []
    config = _BaseSelectionStrategy(
        limit=-1,
        tail_triggers=None,
        maximum_keys_in_memory=1000,
    )
    strat1 = AbstractSelectionStrategy(
        config,
        get_minimal_modyn_config(),
        42,
    )
    assert strat1._pipeline_id == 42
    assert strat1._next_trigger_id == 0

    strat1.trigger()
    assert strat1._next_trigger_id == 1
    strat1.trigger()
    assert strat1._next_trigger_id == 2

    strat2 = AbstractSelectionStrategy(
        config,
        get_minimal_modyn_config(),
        21,
    )
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


def test_store_trigger_num_keys():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.session.add(Trigger(trigger_id=0, pipeline_id=42, num_keys=10, num_partitions=1))
        database.session.commit()

    AbstractSelectionStrategy._store_trigger_num_keys(get_minimal_modyn_config(), 42, 0, 12, 10)

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(TriggerPartition).all()

        assert len(data) == 1
        assert data[0].trigger_id == 0
        assert data[0].pipeline_id == 42
        assert data[0].partition_id == 12
        assert data[0].num_keys == 10
