from __future__ import annotations

from typing import Dict

from modyn.selector.internal.selector_strategies import CoresetStrategy
from modyn.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.utils.utils import flatten, get_partition_for_worker


class Selector:
    """
    This class implements the functionality of the Selector for a concrete pipeline.
    """

    def __init__(
        self, strategy: AbstractSelectionStrategy, pipeline_id: int, num_workers: int, cache_size: int = 100000
    ) -> None:
        self._strategy = strategy
        self._pipeline_id = pipeline_id
        self._num_workers = num_workers

        self._trigger_cache: Dict[int, list[list[tuple[int, float]]]] = {}
        self._maximum_keys_in_cache = cache_size
        self._current_keys_in_cache = 0

        self._trigger_size_cache: Dict[int, int] = {}
        self._trigger_partition_cache: Dict[int, int] = {}

    def get_sample_keys_and_weights(
        self, trigger_id: int, worker_id: int, partition_id: int
    ) -> list[tuple[int, float]]:
        """
        For a given trigger and worker, this function returns the subset of sample
        keys to be queried from storage. It also returns the associated weight of each sample.
        This weight can be used during training to support advanced strategies that want to weight the
        gradient descent step for different samples differently. Explicitly, instead of changing parameters
        by learning_rate * gradient, you would change the parameters by sample_weight * learning_rate * gradient.

        Returns:
            List of tuples for the samples to be returned to that particular worker. The first
            index of the tuple will be the key, and the second index will be that sample's weight.
        """
        if trigger_id not in self._trigger_partition_cache or partition_id >= self._trigger_partition_cache[trigger_id]:
            raise ValueError(f"Invalid request: Trigger {trigger_id}, partition {partition_id}")
        if worker_id < 0 or worker_id >= self._num_workers:
            raise ValueError(f"Asked for worker id {worker_id}, but only have {self._num_workers} workers!")

        if trigger_id in self._trigger_cache:
            start_index, worker_subset_size = get_partition_for_worker(
                worker_id, self._num_workers, len(self._trigger_cache[trigger_id][partition_id])
            )
            training_samples_subset = self._trigger_cache[trigger_id][partition_id][
                start_index : start_index + worker_subset_size
            ]
        else:
            training_samples_subset = self._strategy.get_trigger_partition_keys(
                trigger_id, partition_id, worker_id, self._num_workers
            )

        return training_samples_subset

    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> None:
        self._strategy.inform_data(keys, timestamps, labels)

    def inform_data_and_trigger(self, keys: list[int], timestamps: list[int], labels: list[int]) -> int:
        assert len(keys) == len(timestamps) and len(keys) == len(labels), "Inconsistent list lengths"

        if len(keys) > 0:
            self._strategy.inform_data(keys, timestamps, labels)

        # Calculates the actual training set for that trigger.
        trigger_id, total_keys_in_trigger, partitions_in_trigger = self._strategy.trigger()
        assert trigger_id not in self._trigger_size_cache, "Trigger ID already exists, something went wrong."

        if self._current_keys_in_cache + total_keys_in_trigger <= self._maximum_keys_in_cache:
            # TODO(#178): offer function to delete old triggers from cache, e.g., after a training is done.
            self._trigger_cache[trigger_id] = [
                self._strategy.get_trigger_partition_keys(trigger_id, partition_id)
                for partition_id in range(partitions_in_trigger)
            ]
            self._current_keys_in_cache += total_keys_in_trigger
            assert total_keys_in_trigger == len(
                flatten(self._trigger_cache[trigger_id])
            ), "Inconsistency in DB and Strategy"

        self._trigger_size_cache[trigger_id] = total_keys_in_trigger
        self._trigger_partition_cache[trigger_id] = partitions_in_trigger

        return trigger_id

    def get_number_of_samples(self, trigger_id: int) -> int:
        if trigger_id not in self._trigger_size_cache:
            raise ValueError(f"Trigger ID {trigger_id} does not exist!")

        return self._trigger_size_cache[trigger_id]

    def get_status_bar_scale(self) -> int:
        # the status bar scale is only meaningful if we are using a Downsampling strategy. Otherwise, it's always 100%
        if not isinstance(self._strategy, CoresetStrategy):
            return 100

        return self._strategy.training_status_bar_scale

    def get_number_of_partitions(self, trigger_id: int) -> int:
        if trigger_id not in self._trigger_partition_cache:
            raise ValueError(f"Trigger ID {trigger_id} does not exist!")

        return self._trigger_partition_cache[trigger_id]

    def uses_weights(self) -> bool:
        return self._strategy.uses_weights

    def get_selection_strategy_remote(self) -> tuple[bool, str, dict]:
        if isinstance(self._strategy, CoresetStrategy):
            return (
                self._strategy.requires_remote_computation,
                self._strategy.downsampling_strategy,
                self._strategy.downsampling_params,
            )
        return False, "", {}
