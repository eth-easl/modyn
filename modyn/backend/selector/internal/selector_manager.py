from __future__ import annotations

import json
import logging
from threading import Lock

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.selector import Selector
from modyn.utils.utils import dynamic_module_import

logger = logging.getLogger(__name__)


class SelectorManager:
    def __init__(self, modyn_config: dict) -> None:
        self._modyn_config = modyn_config
        self._selectors: dict[int, Selector] = {}
        self._selector_locks: dict[int, Lock] = {}
        self._next_pipeline_lock = Lock()
        self._selector_cache_size = self._modyn_config["selector"]["keys_in_selector_cache"]

        self.init_metadata_db()

    def init_metadata_db(self) -> None:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            database.create_tables()

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
            with MetadataDatabaseConnection(self._modyn_config) as database:
                pipeline_id = database.register_pipeline(num_workers)

        selection_strategy = self._instantiate_strategy(json.loads(selection_strategy), pipeline_id)
        selector = Selector(selection_strategy, pipeline_id, num_workers, self._selector_cache_size)
        self._selectors[pipeline_id] = selector
        self._selector_locks[pipeline_id] = Lock()
        return pipeline_id

    def get_sample_keys_and_weights(
        self, pipeline_id: int, trigger_id: int, worker_id: int, partition_id: int
    ) -> list[tuple[int, float]]:
        """
        For a given pipeline, trigger, partition of that trigger, and worker, this function returns the subset of sample
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

        return self._selectors[pipeline_id].get_sample_keys_and_weights(trigger_id, worker_id, partition_id)

    def inform_data(self, pipeline_id: int, keys: list[int], timestamps: list[int], labels: list[int]) -> None:
        if pipeline_id not in self._selectors:
            raise ValueError(f"Informing pipeline {pipeline_id} of data. Pipeline does not exist!")

        with self._selector_locks[pipeline_id]:
            self._selectors[pipeline_id].inform_data(keys, timestamps, labels)

    def inform_data_and_trigger(
        self, pipeline_id: int, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> int:
        if pipeline_id not in self._selectors:
            raise ValueError(f"Informing pipeline {pipeline_id} of data and triggering. Pipeline does not exist!")

        with self._selector_locks[pipeline_id]:
            return self._selectors[pipeline_id].inform_data_and_trigger(keys, timestamps, labels)

    def get_number_of_samples(self, pipeline_id: int, trigger_id: int) -> int:
        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested number of samples from pipeline {pipeline_id} which does not exist!")

        return self._selectors[pipeline_id].get_number_of_samples(trigger_id)

    def get_number_of_partitions(self, pipeline_id: int, trigger_id: int) -> int:
        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested number of partitions from pipeline {pipeline_id} which does not exist!")

        return self._selectors[pipeline_id].get_number_of_partitions(trigger_id)

    def _instantiate_strategy(self, selection_strategy: dict, pipeline_id: int) -> AbstractSelectionStrategy:
        strategy_name = selection_strategy["name"]
        maximum_keys_in_memory = selection_strategy["maximum_keys_in_memory"]
        config = selection_strategy["config"] if "config" in selection_strategy else {}
        default_configs = {"limit": -1, "reset_after_trigger": False}

        for setting, default_value in default_configs.items():
            if setting not in config:
                config[setting] = default_value
                logger.warning(f"Setting {setting} to default {default_value} since it was not given.")

        strategy_module = dynamic_module_import("modyn.backend.selector.internal.selector_strategies")
        if not hasattr(strategy_module, strategy_name):
            raise NotImplementedError(f"Strategy {strategy_name} not available!")

        strategy_handler = getattr(strategy_module, strategy_name)

        return strategy_handler(config, self._modyn_config, pipeline_id, maximum_keys_in_memory)
