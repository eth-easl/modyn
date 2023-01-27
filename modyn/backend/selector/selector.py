from __future__ import annotations

from typing import Dict

from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy


class Selector:
    """
    This class implements the functionality of the Selector for a concrete pipeline.
    """

    def __init__(self, strategy: AbstractSelectionStrategy, pipeline_id: int, num_workers: int) -> None:
        self._strategy = strategy
        self._pipeline_id = pipeline_id
        self._num_workers = num_workers

        # To avoid recalculation, for each trigger_id, we cache the key and return the samples,
        # which is a list of tuples (sample_key, sample_weight)
        self._trigger_cache: Dict[int, list[tuple[str, float]]] = {}

    def _get_training_set_partition(
        self, training_samples: list[tuple[str, float]], worker_id: int
    ) -> list[tuple[str, float]]:
        """
        Return the required subset of training samples for the particular worker id
        The subset is calculated by taking an offset from the start based on the given worker id.

        If there is excess data (say there are 14 data points and 5 workers), there are at most
        num_workers extra samples. As such, we make each worker take on one extra, and the final
        worker takes on (probably less) the rest of the data. So we would have the first 4 take
        3 each and the last one takes 2.

        Returns:
            list(tuple(str, float)): the training sample keys for the newly selected training_set
                along with the weight of that sample.
        """
        if worker_id < 0 or worker_id >= self._num_workers:
            raise ValueError(f"Asked for worker id {worker_id}, but only have {self._num_workers} workers!")

        training_set_size = len(training_samples)
        worker_subset_size = int(training_set_size / self._num_workers)

        if training_set_size % self._num_workers > 0:
            worker_subset_size += 1

        start_index = worker_id * worker_subset_size
        training_samples_subset = training_samples[start_index : start_index + worker_subset_size]

        return training_samples_subset

    def get_sample_keys_and_weight(self, trigger_id: int, worker_id: int) -> list[tuple[str, float]]:
        """
        For a give trigger and worker, this function returns the subset of sample
        keys to be queried from storage. It also returns the associated weight of each sample.
        This weight can be used during training to support advanced strategies that want to weight the
        gradient descent step for different samples differently. Explicitly, instead of changing parameters
        by learning_rate * gradient, you would change the parameters by sample_weight * learning_rate * gradient.

        Returns:
            List of tuples for the samples to be returned to that particular worker. The first
            index of the tuple will be the key, and the second index will be that sample's weight.
        """
        if trigger_id not in self._trigger_cache:
            raise ValueError(f"Trigger ID {trigger_id} does not exist!")

        training_samples = self._trigger_cache[trigger_id]
        return self._get_training_set_partition(training_samples, worker_id)

    def inform_data(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        self._strategy.inform_data(keys, timestamps, labels)

    def inform_data_and_trigger(self, keys: list[str], timestamps: list[int], labels: list[int]) -> int:
        self._strategy.inform_data(keys, timestamps, labels)

        # Calculates the actual training set for that trigger.
        trigger_id, data = self._strategy.trigger()
        assert trigger_id not in self._trigger_cache, "Trigger ID already exists, something went wrong."

        self._trigger_cache[trigger_id] = data

        return trigger_id
