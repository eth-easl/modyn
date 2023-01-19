# pylint: disable=no-value-for-parameter
from unittest.mock import patch

import pytest
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.selector import Selector


class MockGRPCHandler:
    def __init__(self, metadata_response):
        self.metadata_response = metadata_response

    def register_training(self, training_set_size, num_workers):  # pylint: disable=unused-argument
        return 5

    def get_samples_by_metadata_query(self, query):  # pylint: disable=unused-argument
        return self.metadata_response

    def get_info_for_training(self, training_id):  # pylint: disable=unused-argument
        return tuple([10, 3])


def noop_constructor_mock(self, config=None, opt=None):  # pylint: disable=unused-argument
    pass


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "__init__", noop_constructor_mock)
@patch.object(Selector, "__init__", noop_constructor_mock)
@patch.object(AbstractSelectionStrategy, "select_new_training_samples")
def test_prepare_training_set(test_select_new_training_samples):
    test_select_new_training_samples.return_value = ["a", "b"]

    selector = Selector(None)
    strategy = AbstractSelectionStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._strategy = strategy
    assert selector._prepare_training_set(0, 0, 0) == ["a", "b"]

    test_select_new_training_samples.return_value = []
    with pytest.raises(ValueError):
        selector._prepare_training_set(0, 0, 3)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(Selector, "__init__", noop_constructor_mock)
@patch.object(AbstractSelectionStrategy, "__init__", noop_constructor_mock)
def test_get_training_set_partition():
    selector = Selector(None)
    strategy = AbstractSelectionStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._strategy = strategy
    selector.grpc = MockGRPCHandler(None)
    strategy._grpc = MockGRPCHandler(None)

    training_samples = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    assert selector._get_training_set_partition(0, training_samples, 0) == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert selector._get_training_set_partition(0, training_samples, 1) == [
        "e",
        "f",
        "g",
        "h",
    ]
    assert selector._get_training_set_partition(0, training_samples, 2) == ["i", "j"]

    with pytest.raises(Exception):
        selector._get_training_set_partition(0, training_samples, 3)
    with pytest.raises(Exception):
        selector._get_training_set_partition(0, training_samples, -1)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "__init__", noop_constructor_mock)
@patch.object(Selector, "__init__", noop_constructor_mock)
@patch.object(Selector, "_prepare_training_set")
def test_get_sample_keys(test__prepare_training_set):
    training_samples = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    training_weights = [1.0] * len(training_samples)
    test__prepare_training_set.return_value = list(zip(training_samples, training_weights))

    selector = Selector()
    strategy = AbstractSelectionStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._strategy = strategy
    selector.grpc = MockGRPCHandler(None)
    strategy._grpc = MockGRPCHandler(None)

    assert selector.get_sample_keys_and_weight(0, 0, 0) == [("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)]
    assert selector.get_sample_keys_and_weight(0, 0, 1) == [("e", 1.0), ("f", 1.0), ("g", 1.0), ("h", 1.0)]
    assert selector.get_sample_keys_and_weight(0, 0, 2) == [("i", 1.0), ("j", 1.0)]
    with pytest.raises(ValueError):
        selector.get_sample_keys_and_weight(0, 0, -1)
    with pytest.raises(ValueError):
        selector.get_sample_keys_and_weight(0, 0, 10)
    with pytest.raises(NotImplementedError):
        selector.select_new_training_samples(0, 0)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(AbstractSelectionStrategy, "__init__", noop_constructor_mock)
@patch.object(Selector, "__init__", noop_constructor_mock)
def test_register_training():
    selector = Selector()
    strategy = AbstractSelectionStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._strategy = strategy
    selector.grpc = MockGRPCHandler(None)
    strategy._grpc = MockGRPCHandler(None)

    assert selector.register_training(1000, 1) == 5
    with pytest.raises(Exception):
        selector.register_training(1000, 0)
    with pytest.raises(Exception):
        selector.register_training(0, 1)
    with pytest.raises(Exception):
        selector.register_training(-1000, 1)
