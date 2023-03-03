# pylint: disable=no-value-for-parameter,redefined-outer-name
from unittest.mock import MagicMock, patch

import pytest
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.selector import Selector


class MockStrategy(AbstractSelectionStrategy):
    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def _on_trigger(self) -> list[tuple[str, float]]:
        return [[]]

    def inform_data(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        pass

    def _reset_state(self) -> None:
        pass


def test_init():
    selec = Selector(MockStrategy(), 42, 2)
    assert selec._pipeline_id == 42
    assert selec._num_workers == 2


def test_get_training_set_partition():
    selector = Selector(MockStrategy(), 42, 3)

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

    with pytest.raises(ValueError):
        selector._get_training_set_partition(training_samples, 3)
    with pytest.raises(ValueError):
        selector._get_training_set_partition(training_samples, -1)


@patch.object(Selector, "_get_training_set_partition")
def test_get_sample_keys_and_weight_cached(test_get_training_set_partition: MagicMock):
    selector = Selector(MockStrategy(), 42, 3)
    selector._trigger_cache[42] = [[("a", 1.0), ("b", 1.0)], [("c", 1.0), ("d", 1.0)]]
    selector._trigger_partition_cache[42] = 2
    test_get_training_set_partition.side_effect = lambda x, y: x

    result = selector.get_sample_keys_and_weights(42, 2, 0)
    assert result == [("a", 1.0), ("b", 1.0)]
    test_get_training_set_partition.assert_called_once_with([("a", 1.0), ("b", 1.0)], 2)

    result = selector.get_sample_keys_and_weights(42, 2, 1)
    assert result == [("c", 1.0), ("d", 1.0)]

    with pytest.raises(ValueError):
        selector.get_sample_keys_and_weights(42, 1337, 0)

    with pytest.raises(ValueError):
        selector.get_sample_keys_and_weights(42, 2, 1337)


@patch.object(Selector, "_get_training_set_partition")
@patch.object(MockStrategy, "get_trigger_partition_keys")
def test_get_sample_keys_and_weight_no_cache(
    test_get_trigger_partition_keys: MagicMock, test_get_training_set_partition: MagicMock
):
    selector = Selector(MockStrategy(), 42, 3)
    selector._trigger_partition_cache[42] = 2
    test_get_training_set_partition.side_effect = lambda x, y: x
    test_get_trigger_partition_keys.return_value = [("a", 1.0), ("b", 1.0)]

    result = selector.get_sample_keys_and_weights(42, 2, 0)
    assert result == [("a", 1.0), ("b", 1.0)]
    test_get_training_set_partition.assert_called_once_with([("a", 1.0), ("b", 1.0)], 2)

    with pytest.raises(ValueError):
        selector.get_sample_keys_and_weights(42, 1337, 0)


@patch.object(MockStrategy, "inform_data")
def test_inform_data(test_inform_data: MagicMock):
    selector = Selector(MockStrategy(), 42, 3)
    selector.inform_data(["a", "b", "c"], [0, 1, 2], ["cat", "dog", "cat"])

    test_inform_data.assert_called_once_with(["a", "b", "c"], [0, 1, 2], ["cat", "dog", "cat"])


@patch.object(MockStrategy, "inform_data")
@patch.object(MockStrategy, "trigger")
@patch.object(MockStrategy, "get_trigger_partition_keys")
def test_inform_data_and_trigger_caching(
    test_get_trigger_partition_keys: MagicMock, test_trigger: MagicMock, test_inform_data: MagicMock
):
    selector = Selector(MockStrategy(), 42, 3)
    assert selector._current_keys_in_cache == 0

    test_trigger.return_value = (42, 2, 2)  # 2 keys in trigger, 2 partitions
    test_get_trigger_partition_keys.return_value = [("a", 1.0)]

    selector._maximum_keys_in_cache = 10  # Enforce that 2 keys fit into cache

    trigger_id = selector.inform_data_and_trigger(["a", "b", "c"], [0, 1, 2], ["cat", "dog", "cat"])

    test_inform_data.assert_called_once_with(["a", "b", "c"], [0, 1, 2], ["cat", "dog", "cat"])
    assert trigger_id == 42
    # We have two partitions with [("a", 1.0)] as data

    # This test configures the selector to store the partitions in memory
    assert selector._trigger_cache[42] == [[("a", 1.0)], [("a", 1.0)]]
    assert selector._trigger_partition_cache[42] == 2


@patch.object(MockStrategy, "inform_data")
@patch.object(MockStrategy, "trigger")
@patch.object(MockStrategy, "get_trigger_partition_keys")
def test_inform_data_and_trigger_nocaching(
    test_get_trigger_partition_keys: MagicMock, test_trigger: MagicMock, test_inform_data: MagicMock
):
    selector = Selector(MockStrategy(), 42, 3)
    assert selector._current_keys_in_cache == 0

    test_trigger.return_value = (42, 2, 2)  # 2 keys in trigger, 2 partitions
    test_get_trigger_partition_keys.return_value = [("a", 1.0)]

    # Enforce that 1 key fit into cache => we can't cache 2 keys
    selector._maximum_keys_in_cache = 1

    trigger_id = selector.inform_data_and_trigger(["a", "b", "c"], [0, 1, 2], ["cat", "dog", "cat"])
    test_inform_data.assert_called_once_with(["a", "b", "c"], [0, 1, 2], ["cat", "dog", "cat"])
    assert trigger_id == 42

    # This test configures the selector such that the partitions do not fit into cache
    assert 42 not in selector._trigger_cache
    assert selector._trigger_size_cache[42] == 2
    assert selector._trigger_partition_cache[42] == 2


def test_get_number_of_samples():
    selector = Selector(MockStrategy(), 42, 3)
    selector._trigger_size_cache[42] = 2

    assert selector.get_number_of_samples(42) == 2

    with pytest.raises(ValueError):
        selector.get_number_of_samples(21)


def test_get_number_of_partitions():
    selector = Selector(MockStrategy(), 42, 3)
    selector._trigger_partition_cache[42] = 2

    assert selector.get_number_of_partitions(42) == 2

    with pytest.raises(ValueError):
        selector.get_number_of_partitions(21)
