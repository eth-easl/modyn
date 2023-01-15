# pylint: disable=no-value-for-parameter
from unittest.mock import patch

import pytest
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.internal.selector_strategies.data_freshness_strategy import DataFreshnessStrategy


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
@patch.object(DataFreshnessStrategy, "__init__", noop_constructor_mock)
@patch.object(DataFreshnessStrategy, "_get_unseen_data")
@patch.object(DataFreshnessStrategy, "_get_seen_data")
def test_base_selector_get_new_training_samples(test__get_seen_data, test__get_unseen_data):
    test__get_unseen_data.return_value = ["a", "b", "c"]
    test__get_seen_data.return_value = ["d"]

    selector = DataFreshnessStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._set_unseen_data_ratio(0.75)
    selector._is_adaptive_ratio = False
    selector._grpc = MockGRPCHandler(None)
    with pytest.raises(Exception):
        selector._set_unseen_data_ratio(1.1)
    with pytest.raises(Exception):
        selector._set_unseen_data_ratio(-0.1)

    assert selector.select_new_training_samples(0, 4) == [
        ("a",),
        ("b",),
        ("c",),
        ("d",),
    ]
    test__get_unseen_data.assert_called_with(0, 3)
    test__get_seen_data.assert_called_with(0, 1)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(DataFreshnessStrategy, "__init__", noop_constructor_mock)
@patch.object(DataFreshnessStrategy, "_get_unseen_data")
@patch.object(DataFreshnessStrategy, "_get_seen_data")
@patch.object(DataFreshnessStrategy, "_get_unseen_data_size")
@patch.object(DataFreshnessStrategy, "_get_seen_data_size")
def test_adaptive_selector_get_new_training_samples(
    test__get_seen_data_size,
    test__get_unseen_data_size,
    test__get_seen_data,
    test__get_unseen_data,
):
    test__get_unseen_data.return_value = ["a"]
    test__get_seen_data.return_value = ["b", "c", "d", "e"]
    test__get_seen_data_size.return_value = 80
    test__get_unseen_data_size.return_value = 20

    selector = DataFreshnessStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._is_adaptive_ratio = True
    selector.unseen_data_ratio = 0.0

    assert selector.select_new_training_samples(0, 5) == [
        ("a",),
        ("b",),
        ("c",),
        ("d",),
        ("e",),
    ]
    test__get_unseen_data.assert_called_with(0, 1)
    test__get_seen_data.assert_called_with(0, 4)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(DataFreshnessStrategy, "__init__", noop_constructor_mock)
def test_base_selector_get_seen_data():
    test_metadata_response = ["a", "b"], [0, 1], [1, 1], [0, 0], ["a", "b"]

    selector = DataFreshnessStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._is_adaptive_ratio = True
    selector._grpc = MockGRPCHandler(test_metadata_response)

    for key in selector._get_seen_data(0, 1):
        assert key in ["a", "b"]

    assert selector._get_seen_data_size(0) == 2


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(DataFreshnessStrategy, "__init__", noop_constructor_mock)
def test_base_selector_get_unseen_data():
    test_metadata_response = ["a", "b"], [0, 1], [0, 0], [0, 0], ["a", "b"]

    selector = DataFreshnessStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._is_adaptive_ratio = True
    selector._grpc = MockGRPCHandler(test_metadata_response)

    for key in selector._get_unseen_data(0, 1):
        assert key in ["a", "b"]

    assert selector._get_unseen_data_size(0) == 2
