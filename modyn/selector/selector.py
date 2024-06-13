from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
from modyn.common.trigger_sample import ArrayWrapper
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models.triggers import Trigger
from modyn.selector.internal.selector_strategies import CoresetStrategy
from modyn.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.utils.utils import flatten, get_partition_for_worker


class Selector:
    """
    This class implements the functionality of the Selector for a concrete pipeline.
    """

    def __init__(
        self,
        strategy: AbstractSelectionStrategy,
        pipeline_id: int,
        num_workers: int,
        modyn_config: dict,
        cache_size: int = 100000,
    ) -> None:
        self._strategy = strategy
        self._pipeline_id = pipeline_id
        self._num_workers = num_workers
        self._modyn_config = modyn_config

        # TODO(#308): Share partition cache between selector instances
        self._trigger_cache: Dict[int, list[ArrayWrapper]] = {}
        self._maximum_keys_in_cache = cache_size
        self._current_keys_in_cache = 0

        self._trigger_size_cache: Dict[int, int] = {}
        self._trigger_partition_cache: Dict[int, int] = {}

    def _populate_trigger_if_exists(self, trigger_id: int) -> None:
        if trigger_id in self._trigger_size_cache:
            assert trigger_id in self._trigger_partition_cache, "Inconsistent state"
            return

        if "metadata_database" not in self._modyn_config:  # Can happen in tests
            return

        with MetadataDatabaseConnection(self._modyn_config) as database:
            trigger: Optional[Trigger] = database.session.get(Trigger, (trigger_id, self._pipeline_id))
            if trigger is None:
                return

            self._trigger_size_cache[trigger_id] = trigger.num_keys
            self._trigger_partition_cache[trigger_id] = trigger.num_partitions

    def get_sample_keys_and_weights(
        self, trigger_id: int, worker_id: int, partition_id: int
    ) -> Union[np.ndarray, ArrayWrapper]:
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
        self._populate_trigger_if_exists(trigger_id)

        if trigger_id not in self._trigger_partition_cache or partition_id >= self._trigger_partition_cache[trigger_id]:
            raise ValueError(f"Invalid request: Trigger {trigger_id}, partition {partition_id}")
        if worker_id < 0 or worker_id >= self._num_workers:
            raise ValueError(f"Asked for worker id {worker_id}, but only have {self._num_workers} workers!")

        if trigger_id in self._trigger_cache:
            start_index, worker_subset_size = get_partition_for_worker(
                worker_id, self._num_workers, len(self._trigger_cache[trigger_id][partition_id])
            )
            return np.asanyarray(
                self._trigger_cache[trigger_id][partition_id][start_index : start_index + worker_subset_size],
                dtype=[("f0", "<i8"), ("f1", "<f8")],
            )
        return self._strategy.get_trigger_partition_keys(trigger_id, partition_id, worker_id, self._num_workers)

    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> dict[str, Any]:
        self._strategy._update_next_trigger_id()
        return self._strategy.inform_data(keys, timestamps, labels)

    def inform_data_and_trigger(
        self, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> tuple[int, dict[str, Any]]:
        assert len(keys) == len(timestamps) and len(keys) == len(labels), "Inconsistent list lengths"
        log: dict[str, Any] = {"cached": False}

        self._strategy._update_next_trigger_id()

        if len(keys) > 0:
            log["inform"] = self._strategy.inform_data(keys, timestamps, labels)

        # Calculates the actual training set for that trigger.
        trigger_id, total_keys_in_trigger, partitions_in_trigger, trigger_log = self._strategy.trigger()
        log["trigger"] = trigger_log

        assert trigger_id not in self._trigger_size_cache, "Trigger ID already exists, something went wrong."

        if self._current_keys_in_cache + total_keys_in_trigger <= self._maximum_keys_in_cache:
            # TODO(#178): offer function to delete old triggers from cache, e.g., after a training is done.
            self._trigger_cache[trigger_id] = [
                self._strategy.get_trigger_partition_keys(trigger_id, partition_id)
                for partition_id in range(partitions_in_trigger)
            ]
            self._current_keys_in_cache += total_keys_in_trigger
            assert total_keys_in_trigger == len(
                flatten(self._trigger_cache[trigger_id])  # type: ignore
            ), "Inconsistency in DB and Strategy"
            log["cached"] = True

        self._trigger_size_cache[trigger_id] = total_keys_in_trigger
        self._trigger_partition_cache[trigger_id] = partitions_in_trigger

        return trigger_id, log

    def get_number_of_samples(self, trigger_id: int) -> int:
        self._populate_trigger_if_exists(trigger_id)

        if trigger_id not in self._trigger_size_cache:
            raise ValueError(f"Trigger ID {trigger_id} does not exist!")

        return self._trigger_size_cache[trigger_id]

    def get_status_bar_scale(self) -> int:
        # the status bar scale is only meaningful if we are using a Downsampling strategy. Otherwise, it's always 100%
        if not isinstance(self._strategy, CoresetStrategy):
            return 100

        return self._strategy.training_status_bar_scale

    def get_number_of_partitions(self, trigger_id: int) -> int:
        self._populate_trigger_if_exists(trigger_id)

        if trigger_id not in self._trigger_partition_cache:
            raise ValueError(f"Trigger ID {trigger_id} does not exist!")

        return self._trigger_partition_cache[trigger_id]

    def get_available_labels(self) -> list[int]:
        return self._strategy.get_available_labels()

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
