from __future__ import annotations

from typing import Dict

from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy


class Selector:
    """
    This class defines the interface of interest, namely the get_sample_keys_and_metadata,
    inform_data, and inform_data_and_trigger methods.
    """

    def __init__(self, strategy: AbstractSelectionStrategy, pipeline_id: int, num_workers: int) -> None:
        self._strategy = strategy
        self._pipeline_id = pipeline_id
        self._num_workers = num_workers
        # The cache will have trigger_id as the key and return the samples,
        # which is a list of tuples (sample_key, sample_weight)
        self._trigger_cache: Dict[int, list[tuple[str, float]]] = {}
        self._current_trigger_id = 0

    def _select_new_training_samples(self, trigger_id: int) -> list[tuple[str, float]]:
        """
        Selects a new training set of samples for the given training id.

        Returns:
            list(tuple(str, float)): the training sample keys for the newly selected training_set
                along with the weight of each sample.
        """
        samples = self._strategy.select_new_training_samples(self._pipeline_id)
        self._trigger_cache[trigger_id] = samples
        return samples

    def _get_training_set(
        self,
        trigger_id: int,
    ) -> list[tuple[str, float]]:
        """
        Get a new training set of samples for the given training id. If this trigger_id
        for a given pipeline_id has been queried before, we get it from cache, otherwise compute it anew.
        We hold the invariant that trigger_id is increasing in time.

        Returns:
            list(tuple(str, float)): the training sample keys for the newly selected training_set
                along with the weight of each sample.
        """
        if trigger_id in self._trigger_cache:
            training_samples = self._trigger_cache[trigger_id]
        else:
            training_samples = self._select_new_training_samples(trigger_id)

        # Throw error if no new samples are selected
        if len(training_samples) == 0:
            raise ValueError(f"No new samples selected for trigger {trigger_id}")

        return training_samples

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
        For a given trigger_id and worker_id, it returns a subset of sample
        keys so that the data can be queried from storage. It also returns the associated weight of each sample.
        This weight can be used during training to support advanced strategies that want to weight the
        gradient descent step for different samples differently. Explicitly, instead of changing parameters
        by learning_rate * gradient, you would change the parameters by sample_weight * learning_rate * gradient.

        Returns:
            List of tuples for the samples to be returned to that particular worker. The first
            index of the tuple will be the key, and the second index will be that sample's weight.
        """
        training_samples = self._get_training_set(trigger_id)
        training_samples_subset = self._get_training_set_partition(training_samples, worker_id)
        return training_samples_subset

    def inform_data(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        self._strategy.inform_data(self._pipeline_id, keys, timestamps, labels)

    def inform_data_and_trigger(self, keys: list[str], timestamps: list[int], labels: list[int]) -> int:
        self._strategy.inform_data(self._pipeline_id, keys, timestamps, labels)
        next_trigger_id = self._current_trigger_id
        self._current_trigger_id += 1
        self._trigger_cache[next_trigger_id] = self._strategy.trigger(self._pipeline_id)
        return next_trigger_id
