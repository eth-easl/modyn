# pylint: disable=no-value-for-parameter,redefined-outer-name,abstract-class-instantiated
import numpy as np
import os
import pathlib
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from modyn.common.trigger_storage_cpp import TriggerSampleStorage
from modyn.metadata_database.metadata_database_connection import (
    MetadataDatabaseConnection,
)
from modyn.metadata_database.models import (
    SelectorStateMetadata,
    Trigger,
    TriggerPartition,
)
from modyn.selector.internal.selector_strategies.abstract_selection_strategy import (
    AbstractSelectionStrategy,
)

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"
TMP_DIR = tempfile.mkdtemp()


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
    strat = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": False},
        get_minimal_modyn_config(),
        42,
        1000,
    )
    assert not strat.has_limit
    assert not strat.reset_after_trigger
    assert strat._pipeline_id == 42
    assert strat._next_trigger_id == 0

    # Test required config check works
    with pytest.raises(ValueError):
        AbstractSelectionStrategy({"limit": -1}, get_minimal_modyn_config(), 42, 1000)

    with pytest.raises(ValueError):
        AbstractSelectionStrategy(
            {"limit": -1, "reset_after_trigger": False},
            get_minimal_modyn_config(),
            42,
            1000,
            ["doesntexist"],
        )

    #  Test reinit works
    strat = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": False},
        get_minimal_modyn_config(),
        42,
        1000,
    )
    strat._next_trigger_id = 1


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test__on_trigger():
    strat = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": False},
        get_minimal_modyn_config(),
        42,
        1000,
    )
    with pytest.raises(NotImplementedError):
        strat._on_trigger()


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test__reset_state():
    strat = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": False},
        get_minimal_modyn_config(),
        42,
        1000,
    )
    with pytest.raises(NotImplementedError):
        strat._reset_state()


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test_inform_data():
    strat = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": False},
        get_minimal_modyn_config(),
        42,
        1000,
    )
    with pytest.raises(NotImplementedError):
        strat.inform_data([], [], [])


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_without_reset(
    test_reset_state: MagicMock, test__on_trigger: MagicMock
):
    strat = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": False},
        get_minimal_modyn_config(),
        42,
        1000,
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

    assert (
        strat.get_trigger_partition_keys(trigger_id, 0)[:]
        == np.array(
            [
                (10, 1.0),
                (11, 1.0),
                (12, 1.0),
            ]
        )
    ).all()


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_without_reset_multiple_partitions(
    test_reset_state: MagicMock, test__on_trigger: MagicMock
):
    strat = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": False},
        get_minimal_modyn_config(),
        42,
        1000,
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

    assert (
        strat.get_trigger_partition_keys(trigger_id, 0)[:]
        == np.array(
            [
                (10, 1.0),
                (11, 1.0),
                (12, 1.0),
            ]
        )
    ).all()
    assert (
        strat.get_trigger_partition_keys(trigger_id, 1)[:]
        == np.array(
            [
                (13, 1.0),
                (14, 1.0),
                (15, 1.0),
            ]
        )
    ).all()

    with pytest.raises(AssertionError):
        strat.get_trigger_partition_keys(trigger_id, 2)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_with_reset(test_reset_state: MagicMock, test__on_trigger: MagicMock):
    strat = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": True}, get_minimal_modyn_config(), 42, 1000
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
    assert strat.get_trigger_partition_keys(trigger_id, 0) == [
        (10, 1.0),
        (11, 1.0),
        (12, 1.0),
    ]


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
@patch.object(AbstractSelectionStrategy, "_reset_state")
def test_trigger_trigger_stored(_: MagicMock, test__on_trigger: MagicMock):
    strat = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": True}, get_minimal_modyn_config(), 42, 1000
    )
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

        data = TriggerSampleStorage(TMP_DIR).get_trigger_samples(
            42,
            0,
            0,
        )

        assert len(data) == 3
        assert data[0][0] == 10
        assert data[0][1] == 1.0
        assert data[1][0] == 11
        assert data[1][1] == 1.0
        assert data[2][0] == 12
        assert data[2][1] == 1.0

        data = TriggerSampleStorage(TMP_DIR).get_trigger_samples(
            42,
            0,
            1,
        )

        assert len(data) == 1
        assert data[0][0] == 13
        assert data[0][1] == 1.0


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test__persist_data():
    strat = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": True}, get_minimal_modyn_config(), 42, 1000
    )
    strat._persist_samples([10, 11, 12], [0, 1, 2], ["dog", "dog", "cat"])

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

        assert keys[0] == 10 and keys[1] == 11 and keys[2] == 12
        assert timestamps[0] == 0 and timestamps[1] == 1 and timestamps[2] == 2
        assert labels[0] == "dog" and labels[1] == "dog" and labels[2] == "cat"


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "_on_trigger")
def test_two_strategies_increase_next_trigger_separately(test__on_trigger: MagicMock):
    test__on_trigger.return_value = []

    strat1 = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": False},
        get_minimal_modyn_config(),
        42,
        1000,
    )
    assert strat1._pipeline_id == 42
    assert strat1._next_trigger_id == 0

    strat1.trigger()
    assert strat1._next_trigger_id == 1
    strat1.trigger()
    assert strat1._next_trigger_id == 2

    strat2 = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": False},
        get_minimal_modyn_config(),
        21,
        1000,
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
        database.session.add(
            Trigger(trigger_id=0, pipeline_id=42, num_keys=10, num_partitions=1)
        )
        database.session.commit()

    AbstractSelectionStrategy._store_trigger_num_keys(
        get_minimal_modyn_config(), 42, 0, 12, 10
    )

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(TriggerPartition).all()

        assert len(data) == 1
        assert data[0].trigger_id == 0
        assert data[0].pipeline_id == 42
        assert data[0].partition_id == 12
        assert data[0].num_keys == 10


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test_get_available_labels_reset():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        # first trigger
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1, sample_key=0, seen_in_trigger_id=0, timestamp=0, label=1
            )
        )
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1, sample_key=1, seen_in_trigger_id=0, timestamp=0, label=18
            )
        )
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1, sample_key=2, seen_in_trigger_id=0, timestamp=0, label=1
            )
        )
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1, sample_key=3, seen_in_trigger_id=0, timestamp=0, label=0
            )
        )
        database.session.commit()

    abstr = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": True}, get_minimal_modyn_config(), 1, 1000
    )
    abstr._next_trigger_id += 1
    assert sorted(abstr.get_available_labels()) == [0, 1, 18]

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        # second trigger
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1, sample_key=4, seen_in_trigger_id=1, timestamp=0, label=0
            )
        )
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1,
                sample_key=5,
                seen_in_trigger_id=1,
                timestamp=0,
                label=890,
            )
        )
        database.session.commit()

    abstr._next_trigger_id += 1
    assert sorted(abstr.get_available_labels()) == [0, 890]


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
def test_get_available_labels_no_reset():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        # first batch of data
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1, sample_key=0, seen_in_trigger_id=0, timestamp=0, label=1
            )
        )
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1, sample_key=1, seen_in_trigger_id=0, timestamp=0, label=18
            )
        )
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1, sample_key=2, seen_in_trigger_id=0, timestamp=0, label=1
            )
        )
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1, sample_key=3, seen_in_trigger_id=0, timestamp=0, label=0
            )
        )
        database.session.commit()

    abstr = AbstractSelectionStrategy(
        {"limit": -1, "reset_after_trigger": False}, get_minimal_modyn_config(), 1, 1000
    )

    assert sorted(abstr.get_available_labels()) == []
    # simulate a trigger
    abstr._next_trigger_id += 1
    assert sorted(abstr.get_available_labels()) == [0, 1, 18]

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        # another batch of data is inserted with just one more class
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1, sample_key=4, seen_in_trigger_id=1, timestamp=0, label=0
            )
        )
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1,
                sample_key=5,
                seen_in_trigger_id=1,
                timestamp=0,
                label=890,
            )
        )
        database.session.commit()

    assert sorted(abstr.get_available_labels()) == [0, 1, 18]
    # simulate a trigger
    abstr._next_trigger_id += 1
    assert sorted(abstr.get_available_labels()) == [0, 1, 18, 890]
