from __future__ import annotations

import json

from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.internal.selector_strategies.data_freshness_strategy import DataFreshnessStrategy
from modyn.backend.selector.selector import Selector


class SelectorManager:
    def __init__(self, modyn_config: dict) -> None:
        self._modyn_config = modyn_config
        self._selectors: dict[int, Selector] = {}
        self._num_workers: dict[int, int] = {}
        self._next_pipeline_id = 0

    def register_pipeline(self, num_workers: int, strategy_configs: str) -> int:
        """
        Creates a new training object in the database with the given num_workers
        Returns:
            The id of the newly created training object
        Throws:
            ValueError if num_workers is not positive.
        """
        if num_workers <= 0:
            raise ValueError(f"Tried to register training with {num_workers} workers.")

        pipeline_id = self._next_pipeline_id
        self._next_pipeline_id += 1

        selection_strategy = self._instantiate_strategy(json.loads(strategy_configs))
        selector = Selector(selection_strategy, pipeline_id, num_workers)
        self._selectors[pipeline_id] = selector
        self._num_workers[pipeline_id] = num_workers
        return pipeline_id

    def get_sample_keys_and_weight(self, pipeline_id: int, trigger_id: int, worker_id: int) -> list[tuple[str, float]]:
        """
        For a given training_id, trigger_id and worker_id, it returns a subset of sample
        keys so that the data can be queried from storage. It also returns the associated weight of each sample.
        This weight can be used during training to support advanced strategies that want to weight the
        gradient descent step for different samples differently. Explicitly, instead of changing parameters
        by learning_rate * gradient, you would change the parameters by sample_weight * learning_rate * gradient.

        Returns:
            List of tuples for the samples to be returned to that particular worker. The first
            index of the tuple will be the key, and the second index will be that sample's weight.
        """
        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested keys from pipeline {pipeline_id} which does not exist!")

        num_workers = self._num_workers[pipeline_id]
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f"Training {pipeline_id} has {num_workers} workers, but queried for worker {worker_id}!")

        return self._selectors[pipeline_id].get_sample_keys_and_weight(trigger_id, worker_id)

    def inform_data(self, pipeline_id: int, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        if pipeline_id not in self._selectors:
            raise ValueError(f"Informing pipeline {pipeline_id} of data. Pipeline does not exist!")
        self._selectors[pipeline_id].inform_data(keys, timestamps, labels)

    def inform_data_and_trigger(
        self, pipeline_id: int, keys: list[str], timestamps: list[int], labels: list[int]
    ) -> int:
        if pipeline_id not in self._selectors:
            raise ValueError(f"Informing pipeline {pipeline_id} of data and triggering. Pipeline does not exist!")
        return self._selectors[pipeline_id].inform_data_and_trigger(keys, timestamps, labels)

    def _instantiate_strategy(self, strategy_configs: dict) -> AbstractSelectionStrategy:
        strategy_name = strategy_configs["name"]
        strategy_config = strategy_configs["configs"]
        if strategy_name == "finetune":
            strategy_config["selector"] = {"unseen_data_ratio": 1.0, "is_adaptive_ratio": False}
            return DataFreshnessStrategy(strategy_config, self._modyn_config)
        raise NotImplementedError(f"{strategy_name} is not implemented")
