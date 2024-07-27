# pylint: disable=no-value-for-parameter,redefined-outer-name
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from modyn.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.selector.selector import Selector


class MockStrategy(AbstractSelectionStrategy):
    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def _init_storage_backend(self) -> AbstractStorageBackend:
        pass

    def _on_trigger(self) -> list[tuple[str, float]]:
        return [[]]

    def inform_data(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        pass

    def _reset_state(self) -> None:
        pass

    def _update_next_trigger_id(self) -> None:
        pass

    def get_available_labels(self) -> list[int]:
        return []


def test_init():
    selec = Selector(MockStrategy(), 42, 2, {})
    assert selec._pipeline_id == 42
    assert selec._num_workers == 2


def test_get_sample_keys_and_weights_cached():
    selector = Selector(MockStrategy(), 42, 3, {})
    selector._trigger_cache[42] = [[(10, 1.0), (11, 1.0)], [(12, 1.0), (13, 1.0)]]
    selector._trigger_partition_cache[42] = 2
    selector._trigger_size_cache[42] = 4

    result = selector.get_sample_keys_and_weights(42, 0, 0)
    assert result == np.array([(10, 1.0)], dtype=[("f0", "<i8"), ("f1", "<f8")])

    result = selector.get_sample_keys_and_weights(42, 0, 1)
    assert result == np.array([(12, 1.0)], dtype=[("f0", "<i8"), ("f1", "<f8")])

    with pytest.raises(ValueError):
        selector.get_sample_keys_and_weights(42, 1337, 0)

    with pytest.raises(ValueError):
        selector.get_sample_keys_and_weights(42, 2, 1337)


@patch.object(MockStrategy, "get_trigger_partition_keys")
def test_get_sample_keys_and_weight_no_cache(test_get_trigger_partition_keys: MagicMock):
    selector = Selector(MockStrategy(), 42, 3, {})
    selector._trigger_partition_cache[42] = 2
    test_get_trigger_partition_keys.return_value = [(10, 1.0), (11, 1.0)]
    selector._trigger_size_cache[42] = 2

    result = selector.get_sample_keys_and_weights(42, 2, 0)
    assert result == [(10, 1.0), (11, 1.0)]

    with pytest.raises(ValueError):
        selector.get_sample_keys_and_weights(42, 1337, 0)


@patch.object(MockStrategy, "inform_data")
def test_inform_data(test_inform_data: MagicMock):
    selector = Selector(MockStrategy(), 42, 3, {})
    selector.inform_data([10, 11, 12], [0, 1, 2], ["cat", "dog", "cat"])

    test_inform_data.assert_called_once_with([10, 11, 12], [0, 1, 2], ["cat", "dog", "cat"])


@patch.object(MockStrategy, "inform_data")
@patch.object(MockStrategy, "trigger")
@patch.object(MockStrategy, "get_trigger_partition_keys")
def test_inform_data_and_trigger_caching(
    test_get_trigger_partition_keys: MagicMock, test_trigger: MagicMock, test_inform_data: MagicMock
):
    selector = Selector(MockStrategy(), 42, 3, {})
    assert selector._current_keys_in_cache == 0

    test_trigger.return_value = (42, 2, 2, {})  # 2 keys in trigger, 2 partitions
    test_get_trigger_partition_keys.return_value = [(10, 1.0)]

    selector._maximum_keys_in_cache = 10  # Enforce that 2 keys fit into cache

    trigger_id, _ = selector.inform_data_and_trigger([10, 11, 12], [0, 1, 2], ["cat", "dog", "cat"])

    test_inform_data.assert_called_once_with([10, 11, 12], [0, 1, 2], ["cat", "dog", "cat"])
    assert trigger_id == 42
    # We have two partitions with [(10, 1.0)] as data

    # This test configures the selector to store the partitions in memory
    assert selector._trigger_cache[42] == [[(10, 1.0)], [(10, 1.0)]]
    assert selector._trigger_partition_cache[42] == 2
    assert selector._trigger_size_cache[42] == 2


@patch.object(MockStrategy, "inform_data")
@patch.object(MockStrategy, "trigger")
@patch.object(MockStrategy, "get_trigger_partition_keys")
def test_inform_data_and_trigger_nocaching(
    test_get_trigger_partition_keys: MagicMock, test_trigger: MagicMock, test_inform_data: MagicMock
):
    selector = Selector(MockStrategy(), 42, 3, {})
    assert selector._current_keys_in_cache == 0

    test_trigger.return_value = (42, 2, 2, {})  # 2 keys in trigger, 2 partitions
    test_get_trigger_partition_keys.return_value = [(10, 1.0)]

    # Enforce that 1 key fit into cache => we can't cache 2 keys
    selector._maximum_keys_in_cache = 1

    trigger_id, _ = selector.inform_data_and_trigger([10, 11, 12], [0, 1, 2], ["cat", "dog", "cat"])
    test_inform_data.assert_called_once_with([10, 11, 12], [0, 1, 2], ["cat", "dog", "cat"])
    assert trigger_id == 42

    # This test configures the selector such that the partitions do not fit into cache
    assert 42 not in selector._trigger_cache
    assert selector._trigger_size_cache[42] == 2
    assert selector._trigger_partition_cache[42] == 2


def test_get_number_of_samples():
    selector = Selector(MockStrategy(), 42, 3, {})
    selector._trigger_size_cache[42] = 2
    selector._trigger_partition_cache[42] = 1

    assert selector.get_number_of_samples(42) == 2

    with pytest.raises(ValueError):
        selector.get_number_of_samples(21)


def test_get_number_of_partitions():
    selector = Selector(MockStrategy(), 42, 3, {})
    selector._trigger_partition_cache[42] = 2
    selector._trigger_size_cache[42] = 2

    assert selector.get_number_of_partitions(42) == 2

    with pytest.raises(ValueError):
        selector.get_number_of_partitions(21)
