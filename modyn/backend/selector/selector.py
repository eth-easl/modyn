from __future__ import annotations

from typing import Dict

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.internal.selector_strategies.data_freshness_strategy import DataFreshnessStrategy


class Selector:
    """
    This class defines the interface of interest, namely the get_sample_keys_and_metadata method.
    """

    def __init__(self, modyn_config: dict, pipeline_config: dict) -> None:
        self._modyn_config = modyn_config
        self._strategy = self._get_strategy(pipeline_config)
        # The cache will have training_id, training_set_number as the key and return the samples,
        # which is a list of tuples (sample_key, sample_weight)
        self._training_samples_cache: Dict[tuple[int, int], list[tuple[str, float]]] = {}

    def _select_new_training_samples(self, training_id: int, training_set_number: int) -> list[tuple[str, float]]:
        """
        Selects a new training set of samples for the given training id.

        Returns:
            list(tuple(str, float)): the training sample keys for the newly selected training_set
                along with the weight of each sample.
        """
        samples = self._strategy.select_new_training_samples(training_id)
        self._training_samples_cache[(training_id, training_set_number)] = samples
        return samples

    def _get_training_set(
        self,
        training_id: int,
        training_set_number: int,
    ) -> list[tuple[str, float]]:
        """
        Get a new training set of samples for the given training id. If this training_set_number
        for a given training_id has been queried before, we get it from cache, otherwise compute it anew
        and remove from the cache any previous training sets for this training_id. We expect that
        training_set_number is increasing in time.

        Returns:
            list(tuple(str, float)): the training sample keys for the newly selected training_set
                along with the weight of each sample.
        """
        if (training_id, training_set_number) in self._training_samples_cache:
            training_samples = self._training_samples_cache[(training_id, training_set_number)]
        else:
            training_samples = self._select_new_training_samples(training_id, training_set_number)

        # Throw error if no new samples are selected
        if len(training_samples) == 0:
            raise ValueError(f"No new samples selected for training set {training_set_number}")

        return training_samples

    def _get_training_set_partition(
        self, training_id: int, training_samples: list[tuple[str, float]], worker_id: int
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
        with MetadataDatabaseConnection(self._modyn_config) as database:
            num_workers = database.get_training_information(training_id)

        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f"Asked for worker id {worker_id}, but only have {num_workers} workers!")

        training_set_size = len(training_samples)
        worker_subset_size = int(training_set_size / num_workers)
        if training_set_size % num_workers > 0:
            worker_subset_size += 1
        start_index = worker_id * worker_subset_size
        training_samples_subset = training_samples[start_index : start_index + worker_subset_size]
        return training_samples_subset

    def register_training(self, num_workers: int) -> int:
        """
        Creates a new training object in the database with the given num_workers
        Returns:
            The id of the newly created training object
        Throws:
            ValueError if num_workers is not positive.
        """
        if num_workers <= 0:
            raise ValueError(f"Tried to register training with {num_workers} workers.")

        with MetadataDatabaseConnection(self._modyn_config) as database:
            return database.register_training(num_workers)

    def get_sample_keys_and_weight(
        self, training_id: int, training_set_number: int, worker_id: int
    ) -> list[tuple[str, float]]:
        """
        For a given training_id, training_set_number and worker_id, it returns a subset of sample
        keys so that the data can be queried from storage. It also returns the associated weight of each sample.
        This weight can be used during training to support advanced strategies that want to weight the
        gradient descent step for different samples differently. Explicitly, instead of changing parameters
        by learning_rate * gradient, you would change the parameters by sample_weight * learning_rate * gradient.

        Returns:
            List of tuples for the samples to be returned to that particular worker. The first
            index of the tuple will be the key, and the second index will be that sample's weight.
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            num_workers = database.get_training_information(training_id)

        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f"Training {training_id} has {num_workers} workers, but queried for worker {worker_id}!")

        training_samples = self._get_training_set(training_id, training_set_number)
        training_samples_subset = self._get_training_set_partition(training_id, training_samples, worker_id)
        return training_samples_subset

    def _get_strategy(self, pipeline_config: dict) -> AbstractSelectionStrategy:
        strategy_name = pipeline_config["training"]["strategy"]
        strategy_config = pipeline_config["training"]["strategy_config"]
        if strategy_name == "finetune":
            strategy_config["selector"] = {"unseen_data_ratio": 1.0, "is_adaptive_ratio": False}
            return DataFreshnessStrategy(strategy_config, self._modyn_config)
        raise NotImplementedError(f"{strategy_name} is not implemented")
