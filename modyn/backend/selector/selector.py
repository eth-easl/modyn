from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.internal.grpc.grpc_handler import GRPCHandler

class Selector:
    """
    This class defines the interface of interest, namely the .
    """

    def __init__(self, strategy: AbstractSelectionStrategy, modyn_config: dict) -> None:
        self.grpc = GRPCHandler(modyn_config)
        self._strategy = strategy

    def select_new_training_samples(self, training_id: int, training_set_size: int) -> list[tuple[str, ...]]:
        """
        Selects a new training set of samples for the given training id.

        Returns:
            list(tuple(str, ...)): the training sample keys for the newly selected training_set with a variable
                       number of auxiliary data (concrete typing in subclasses defined)
        """
        return self._strategy._select_new_training_samples(training_id, training_set_size)

    def _prepare_training_set(
        self,
        training_id: int,
        training_set_number: int,
        training_set_size: int,
    ) -> list[tuple[str, ...]]:
        """
        Create a new training set of samples for the given training id. New samples are selected from
        the select_new_samples method and are inserted into the database for the given set number.

        Returns:
            list(tuple(str, ...)): the training sample keys for the newly prepared training_set with a variable
                       number of auxiliary data (concrete typing in subclasses defined)
        """
        training_samples = self.select_new_training_samples(training_id, training_set_size)

        # Throw error if no new samples are selected
        if len(training_samples) == 0:
            raise ValueError(f"No new samples selected for training set {training_set_number}")

        return training_samples

    def _get_training_set_partition(
        self, training_id: int, training_samples: list[tuple[str, ...]], worker_id: int
    ) -> list[tuple[str, ...]]:
        """
        Return the required subset of training samples for the particular worker id
        The subset is calculated by taking an offset from the start based on the given worker id

        Returns:
            list(tuple(str, ...)): the training sample keys for the newly prepared training_set for
                                   the particular worker id with a variable number of auxiliary data
                                   (concrete typing in subclasses defined)
        """
        training_set_size, num_workers = self.grpc.get_info_for_training(training_id)

        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f"Asked for worker id {worker_id}, but only have {num_workers} workers!")

        worker_subset_size = int(training_set_size / num_workers)
        if training_set_size % num_workers > 0:
            worker_subset_size += 1
        start_index = worker_id * worker_subset_size
        training_samples_subset = training_samples[
            start_index : min(start_index + worker_subset_size, len(training_samples))
        ]
        return training_samples_subset

    def register_training(self, training_set_size: int, num_workers: int) -> int:
        """
        Creates a new training object in the database with the given training_set_size and num_workers
        Returns:
            The id of the newly created training object
        Throws:
            ValueError if training_set_size or num_workers is not positive.
        """
        if num_workers <= 0 or training_set_size <= 0:
            raise ValueError(
                f"Tried to register training with {num_workers} workers and {training_set_size} data points."
            )

        return self.grpc.register_training(training_set_size, num_workers)

    def get_sample_keys(self, training_id: int, training_set_number: int, worker_id: int) -> list[tuple[str, ...]]:
        """
        For a given training_id, training_set_number and worker_id, it returns a subset of sample
        keys so that the data can be queried from storage.

        Returns:
            List of tuples for the samples to be returned to that particular worker. The first
            index of the tuple will be the key, along with auxiliary data defined in the concrete subclass.
        """
        training_set_size, num_workers = self.grpc.get_info_for_training(training_id)
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f"Training {training_id} has {num_workers} workers, but queried for worker {worker_id}!")

        training_samples = self._prepare_training_set(training_id, training_set_number, training_set_size)

        training_samples_subset = self._get_training_set_partition(training_id, training_samples, worker_id)

        return training_samples_subset
