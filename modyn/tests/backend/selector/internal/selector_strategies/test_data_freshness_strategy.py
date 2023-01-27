# pylint: disable=no-value-for-parameter
import os
import pathlib
from unittest.mock import patch

import pytest
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.metadata_database.models.training import Training
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.internal.selector_strategies.freshness_sampling_strategy import FreshnessSamplingStrategy

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


def noop_constructor_mock(self, config=None, opt=None):  # pylint: disable=unused-argument
    self._modyn_config = get_minimal_modyn_config()


def setup():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()

        training = Training(1)
        database.session.add(training)
        database.session.commit()

        metadata = Metadata("test_key", 100, 0.5, False, 1, b"test_data", training.training_id)

        metadata.metadata_id = 1  # SQLite does not support autoincrement for composite primary keys
        database.session.add(metadata)

        metadata2 = Metadata("test_key2", 101, 0.75, True, 2, b"test_data2", training.training_id)

        metadata2.metadata_id = 2  # SQLite does not support autoincrement for composite primary keys
        database.session.add(metadata2)

        database.session.commit()


def teardown():
    os.remove(database_path)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(FreshnessSamplingStrategy, "__init__", noop_constructor_mock)
@patch.object(FreshnessSamplingStrategy, "_get_unseen_data")
@patch.object(FreshnessSamplingStrategy, "_get_seen_data")
def test_base_selector_with_limit_get_new_training_samples(test__get_seen_data, test__get_unseen_data):
    test__get_unseen_data.return_value = ["a", "b", "c"]
    test__get_seen_data.return_value = ["d"]

    selector = FreshnessSamplingStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._set_unseen_data_ratio(0.75)
    selector.training_set_size_limit = 4
    selector._is_adaptive_ratio = False
    with pytest.raises(Exception):
        selector._set_unseen_data_ratio(1.1)
    with pytest.raises(Exception):
        selector._set_unseen_data_ratio(-0.1)

    assert selector._on_trigger(0) == [
        ("a", 1.0),
        ("b", 1.0),
        ("c", 1.0),
        ("d", 1.0),
    ]
    test__get_unseen_data.assert_called_with(0, 3)
    test__get_seen_data.assert_called_with(0, 1)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(FreshnessSamplingStrategy, "__init__", noop_constructor_mock)
@patch.object(FreshnessSamplingStrategy, "_get_unseen_data")
@patch.object(FreshnessSamplingStrategy, "_get_seen_data")
@patch.object(FreshnessSamplingStrategy, "_get_unseen_data_size")
@patch.object(FreshnessSamplingStrategy, "_get_seen_data_size")
def test_adaptive_selector_with_limit_get_new_training_samples(
    test__get_seen_data_size,
    test__get_unseen_data_size,
    test__get_seen_data,
    test__get_unseen_data,
):
    test__get_unseen_data.return_value = ["a"]
    test__get_seen_data.return_value = ["b", "c", "d", "e"]
    test__get_seen_data_size.return_value = 4
    test__get_unseen_data_size.return_value = 1

    selector = FreshnessSamplingStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector.training_set_size_limit = 5
    selector._is_adaptive_ratio = True
    selector.unseen_data_ratio = 0.0

    assert selector._on_trigger(0) == [
        ("a", 1.0),
        ("b", 1.0),
        ("c", 1.0),
        ("d", 1.0),
        ("e", 1.0),
    ]
    test__get_unseen_data.assert_called_with(0, 1)
    test__get_seen_data.assert_called_with(0, 4)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(FreshnessSamplingStrategy, "__init__", noop_constructor_mock)
@patch.object(FreshnessSamplingStrategy, "_get_unseen_data")
@patch.object(FreshnessSamplingStrategy, "_get_seen_data")
@patch.object(FreshnessSamplingStrategy, "_get_unseen_data_size")
@patch.object(FreshnessSamplingStrategy, "_get_seen_data_size")
def test_base_selector_without_limit_get_new_training_samples(
    test__get_seen_data_size, test__get_unseen_data_size, test__get_seen_data, test__get_unseen_data
):
    test__get_unseen_data.return_value = ["a", "b", "c"]
    test__get_seen_data.return_value = ["d"]
    test__get_seen_data_size.return_value = 1
    test__get_unseen_data_size.return_value = 4

    selector = FreshnessSamplingStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._set_unseen_data_ratio(0.75)
    selector.training_set_size_limit = -1
    selector._is_adaptive_ratio = False
    with pytest.raises(Exception):
        selector._set_unseen_data_ratio(1.1)
    with pytest.raises(Exception):
        selector._set_unseen_data_ratio(-0.1)

    assert selector._on_trigger(0) == [
        ("a", 1.0),
        ("b", 1.0),
        ("c", 1.0),
        ("d", 1.0),
    ]
    test__get_unseen_data.assert_called_with(0, 3)
    test__get_seen_data.assert_called_with(0, -1)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(FreshnessSamplingStrategy, "__init__", noop_constructor_mock)
@patch.object(FreshnessSamplingStrategy, "_get_unseen_data")
@patch.object(FreshnessSamplingStrategy, "_get_seen_data")
@patch.object(FreshnessSamplingStrategy, "_get_unseen_data_size")
@patch.object(FreshnessSamplingStrategy, "_get_seen_data_size")
def test_adaptive_selector_without_limit_get_new_training_samples(
    test__get_seen_data_size,
    test__get_unseen_data_size,
    test__get_seen_data,
    test__get_unseen_data,
):
    test__get_unseen_data.return_value = ["a"]
    test__get_seen_data.return_value = ["b", "c", "d", "e"]
    test__get_seen_data_size.return_value = 4
    test__get_unseen_data_size.return_value = 1

    selector = FreshnessSamplingStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector.training_set_size_limit = -1
    selector._is_adaptive_ratio = True
    selector.unseen_data_ratio = 0.0

    assert selector._on_trigger(0) == [
        ("a", 1.0),
        ("b", 1.0),
        ("c", 1.0),
        ("d", 1.0),
        ("e", 1.0),
    ]
    test__get_unseen_data.assert_called_with(0, -1)
    test__get_seen_data.assert_called_with(0, -1)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(FreshnessSamplingStrategy, "__init__", noop_constructor_mock)
def test_base_selector_get_seen_data():
    selector = FreshnessSamplingStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._is_adaptive_ratio = True

    for key in selector._get_seen_data(1, 1):
        assert key in ["test_key2"]

    assert selector._get_seen_data_size(1) == 1


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(FreshnessSamplingStrategy, "__init__", noop_constructor_mock)
def test_base_selector_get_unseen_data():
    selector = FreshnessSamplingStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._is_adaptive_ratio = True

    for key in selector._get_unseen_data(1, 1):
        assert key in ["test_key"]

    assert selector._get_unseen_data_size(1) == 1
