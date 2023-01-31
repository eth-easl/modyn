from __future__ import annotations

import json
import logging
from threading import Lock

from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.internal.selector_strategies.freshness_sampling_strategy import FreshnessSamplingStrategy
from modyn.backend.selector.selector import Selector

logger = logging.getLogger(__name__)


class SelectorManager:
    def __init__(self, modyn_config: dict) -> None:
        self._modyn_config = modyn_config
        self._selectors: dict[int, Selector] = {}
        self._selector_locks: dict[int, Lock] = {}
        self._next_pipeline_id = 0
        self._next_pipeline_lock = Lock()

    def register_pipeline(self, num_workers: int, selection_strategy: str) -> int:
        """
        Registers a new pipeline at the Selector.
        Returns:
            The id of the newly created training object
        Throws:
            ValueError if num_workers is not positive.
        """
        if num_workers < 0:
            raise ValueError(f"Tried to register training with {num_workers} workers.")

        with self._next_pipeline_lock:
            pipeline_id = self._next_pipeline_id
            self._next_pipeline_id += 1

        selection_strategy = self._instantiate_strategy(json.loads(selection_strategy), pipeline_id)
        selector = Selector(selection_strategy, pipeline_id, num_workers)
        self._selectors[pipeline_id] = selector
        self._selector_locks[pipeline_id] = Lock()
        return pipeline_id

    def get_sample_keys_and_weights(self, pipeline_id: int, trigger_id: int, worker_id: int) -> list[tuple[str, float]]:
        """
        For a given pipeline, trigger and worker, this function returns the subset of sample
        keys to be queried from storage. It also returns the associated weight of each sample.
        This weight can be used during training to support advanced strategies that want to weight the
        gradient descent step for different samples differently. Explicitly, instead of changing parameters
        by learning_rate * gradient, you would change the parameters by sample_weight * learning_rate * gradient.

        Returns:
            List of tuples for the samples to be returned to that particular worker. The first
            index of the tuple will be the key, and the second index will be that sample's weight.
        """
        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested keys from pipeline {pipeline_id} which does not exist!")

        num_workers = self._selectors[pipeline_id]._num_workers
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f"Training {pipeline_id} has {num_workers} workers, but queried for worker {worker_id}!")

        return self._selectors[pipeline_id].get_sample_keys_and_weights(trigger_id, worker_id)

    def inform_data(self, pipeline_id: int, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        if pipeline_id not in self._selectors:
            raise ValueError(f"Informing pipeline {pipeline_id} of data. Pipeline does not exist!")

        with self._selector_locks[pipeline_id]:
            self._selectors[pipeline_id].inform_data(keys, timestamps, labels)

    def inform_data_and_trigger(
        self, pipeline_id: int, keys: list[str], timestamps: list[int], labels: list[int]
    ) -> int:
        if pipeline_id not in self._selectors:
            raise ValueError(f"Informing pipeline {pipeline_id} of data and triggering. Pipeline does not exist!")

        with self._selector_locks[pipeline_id]:
            return self._selectors[pipeline_id].inform_data_and_trigger(keys, timestamps, labels)

    def _instantiate_strategy(self, selection_strategy: dict, pipeline_id: int) -> AbstractSelectionStrategy:
        strategy_name = selection_strategy["name"]
        config = selection_strategy["config"] if "config" in selection_strategy else {}
        assert "configs" not in selection_strategy, "Found legacy usage of 'configs'"

        if "limit" not in config:
            config["limit"] = -1
            logger.warning("No explicit limit given, disabling.")

        if "reset_after_trigger" not in config:
            config["reset_after_trigger"] = False
            logger.warning("No reset setting given, disabling reset.")

        if strategy_name == "finetune":
            config["unseen_data_ratio"] = 1.0
            config["is_adaptive_ratio"] = False
            # TODO(MaxiBoether): switch to newdatastrategy and remove adaptive stuff
            return FreshnessSamplingStrategy(config, self._modyn_config, pipeline_id)

        raise NotImplementedError(f"{strategy_name} is not supported")
