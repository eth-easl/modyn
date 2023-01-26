# pylint: disable=no-value-for-parameter
import os
import pathlib
from unittest.mock import patch

import pytest
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.selector import Selector

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


class MockStrategy:
    def __init__(self, desired_result):
        self.result = desired_result
        self.times_called = 0

    def select_new_training_samples(self, pipeline_id):  # pylint: disable=unused-argument
        self.times_called += 1
        return self.result


def noop_constructor_mock(self, config=None, opt=None):  # pylint: disable=unused-argument
    self._modyn_config = get_minimal_modyn_config()


def setup():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        database.register_pipeline(3)


def teardown():
    os.remove(database_path)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "__init__", noop_constructor_mock)
@patch.object(Selector, "__init__", noop_constructor_mock)
@patch.object(Selector, "_select_new_training_samples")
def test_get_training_set(test__select_new_training_samples):
    test__select_new_training_samples.return_value = ["a", "b"]

    selector = Selector(None)
    strategy = AbstractSelectionStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._strategy = strategy
    selector._training_samples_cache = {}
    assert selector._get_training_set(0) == ["a", "b"]

    test__select_new_training_samples.return_value = []
    with pytest.raises(ValueError):
        selector._get_training_set(3)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(Selector, "__init__", noop_constructor_mock)
@patch.object(AbstractSelectionStrategy, "__init__", noop_constructor_mock)
def test_get_training_set_partition():
    selector = Selector(None)
    strategy = AbstractSelectionStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._strategy = strategy
    selector._num_workers = 3
    selector._pipeline_id = 0

    training_samples = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    assert selector._get_training_set_partition(training_samples, 0) == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert selector._get_training_set_partition(training_samples, 1) == [
        "e",
        "f",
        "g",
        "h",
    ]
    assert selector._get_training_set_partition(training_samples, 2) == ["i", "j"]

    with pytest.raises(Exception):
        selector._get_training_set_partition(training_samples, 3)
    with pytest.raises(Exception):
        selector._get_training_set_partition(training_samples, -1)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "__init__", noop_constructor_mock)
@patch.object(Selector, "__init__", noop_constructor_mock)
@patch.object(Selector, "_get_training_set")
def test_get_sample_keys(test__get_training_set):
    training_samples = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    training_weights = [1.0] * len(training_samples)
    test__get_training_set.return_value = list(zip(training_samples, training_weights))

    selector = Selector()
    strategy = AbstractSelectionStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._strategy = strategy
    selector._num_workers = 3
    selector._pipeline_id = 0

    assert selector.get_sample_keys_and_weight(0, 0) == [("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)]
    assert selector.get_sample_keys_and_weight(0, 1) == [("e", 1.0), ("f", 1.0), ("g", 1.0), ("h", 1.0)]
    assert selector.get_sample_keys_and_weight(0, 2) == [("i", 1.0), ("j", 1.0)]
    with pytest.raises(ValueError):
        selector.get_sample_keys_and_weight(0, -1)
    with pytest.raises(ValueError):
        selector.get_sample_keys_and_weight(0, 10)


@patch.object(Selector, "__init__", noop_constructor_mock)
def test_select_new_training_samples_caching():
    samples_1 = [("a", 1.0), ("b", 1.0)]
    samples_2 = [("c", 1.0), ("d", 1.0)]

    selector = Selector()
    selector._training_samples_cache = {}
    selector._num_workers = 3
    selector._pipeline_id = 0
    selector._strategy = MockStrategy(desired_result=samples_1)

    # I want to test the following. First call it twice with training set number 0.
    # Assert that you get the right answer each time, and _strategy is called only once.
    # Then, switch it to a next set, training set number 1. Repeat the process.
    assert selector._strategy.times_called == 0
    assert selector._get_training_set(0) == samples_1
    assert selector._strategy.times_called == 1
    assert selector._get_training_set(0) == samples_1
    assert selector._strategy.times_called == 1

    selector._strategy = MockStrategy(desired_result=samples_2)

    assert selector._get_training_set(1) == samples_2
    assert len(selector._training_samples_cache.keys()) == 2
    assert selector._strategy.times_called == 1
    assert selector._get_training_set(1) == samples_2
    assert selector._strategy.times_called == 1
