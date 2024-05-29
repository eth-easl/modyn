from __future__ import annotations

import logging
import os
import shutil
from multiprocessing import Manager
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, Optional

from modyn.config import SelectionStrategy as SelectionStrategyModel
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models.pipelines import Pipeline
from modyn.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.selector.selector import Selector
from modyn.utils.utils import dynamic_module_import, is_directory_writable
from pydantic import TypeAdapter

logger = logging.getLogger(__name__)


class SelectorManager:
    def __init__(self, modyn_config: dict) -> None:
        self._modyn_config = modyn_config
        self._manager = Manager()
        self._selectors: dict[int, Selector] = {}
        self._selector_locks: DictProxy[int, Any] = self._manager.dict()
        self._selector_cache_size = self._modyn_config["selector"]["keys_in_selector_cache"]

        # TODO(309): currently we have to prepare N locks and then share.
        # This is because we cannot share the manager with subprocesses.
        # For now not a big problem since we mostly run one pipeline but we might want to redesign this.
        self._prepared_locks = [self._manager.Lock() for _ in range(64)]

        self.init_metadata_db()
        self._init_trigger_sample_directory()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_manager"]
        return state

    def init_metadata_db(self) -> None:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            database.create_tables()

    def _init_trigger_sample_directory(self) -> None:
        ignore_existing_trigger_samples = (
            self._modyn_config["selector"]["ignore_existing_trigger_samples"]
            if "ignore_existing_trigger_samples" in self._modyn_config["selector"]
            else False
        )
        trigger_sample_directory = self._modyn_config["selector"]["trigger_sample_directory"]

        if not Path(trigger_sample_directory).exists():
            raise ValueError(
                f"The trigger sample directory {trigger_sample_directory} does not exist. \
                  Please create the directory or mount another, existing directory."
            )

        if any(Path(trigger_sample_directory).iterdir()) and not ignore_existing_trigger_samples:
            raise ValueError(
                f"The trigger sample directory {trigger_sample_directory} is not empty. \
                  Please delete the directory or set the ignore_existing_trigger_samples flag to True."
            )

        if not is_directory_writable(Path(trigger_sample_directory)):
            raise ValueError(
                f"The trigger sample directory {trigger_sample_directory} is not writable. \
                  Please check the directory permissions and try again.\n"
                + f"Directory info: {os.stat(trigger_sample_directory)}"
            )

    def _populate_pipeline_if_exists(self, pipeline_id: int) -> None:
        if pipeline_id in self._selectors:
            return

        with MetadataDatabaseConnection(self._modyn_config) as database:
            pipeline: Optional[Pipeline] = database.session.get(Pipeline, pipeline_id)
            if pipeline is None:
                return
            logging.info(
                "[%d] Instantiating new selector for pipeline %d"
                + " that was in the DB but previously unknown to this process",
                os.getpid(),
                pipeline_id,
            )
            self._selector_locks[pipeline_id] = self._prepared_locks[pipeline_id % len(self._prepared_locks)]

            selection_strategy = TypeAdapter(SelectionStrategyModel).validate_json(pipeline.selection_strategy)
            self._instantiate_selector(pipeline_id, pipeline.num_workers, selection_strategy)

    def _instantiate_selector(
        self, pipeline_id: int, num_workers: int, selection_strategy: SelectionStrategyModel
    ) -> None:
        assert pipeline_id in self._selector_locks, f"Trying to register pipeline {pipeline_id} without existing lock!"
        parsed_selection_strategy = self._instantiate_strategy(selection_strategy, pipeline_id)
        selector = Selector(
            parsed_selection_strategy, pipeline_id, num_workers, self._modyn_config, self._selector_cache_size
        )
        self._selectors[pipeline_id] = selector

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
        self._populate_pipeline_if_exists(pipeline_id)

        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested keys from pipeline {pipeline_id} which does not exist!")

        num_workers = self._selectors[pipeline_id]._num_workers
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f"Training {pipeline_id} has {num_workers} workers, but queried for worker {worker_id}!")

        return self._selectors[pipeline_id].get_sample_keys_and_weights(
            trigger_id, worker_id, partition_id
        )  # type: ignore

    def inform_data(
        self, pipeline_id: int, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> dict[str, object]:
        self._populate_pipeline_if_exists(pipeline_id)

        if pipeline_id not in self._selectors:
            raise ValueError(f"Informing pipeline {pipeline_id} of data. Pipeline does not exist!")

        with self._selector_locks[pipeline_id]:
            return self._selectors[pipeline_id].inform_data(keys, timestamps, labels)

    def inform_data_and_trigger(
        self, pipeline_id: int, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> tuple[int, dict[str, object]]:
        self._populate_pipeline_if_exists(pipeline_id)

        if pipeline_id not in self._selectors:
            raise ValueError(f"Informing pipeline {pipeline_id} of data and triggering. Pipeline does not exist!")

        with self._selector_locks[pipeline_id]:
            return self._selectors[pipeline_id].inform_data_and_trigger(keys, timestamps, labels)

    def get_number_of_samples(self, pipeline_id: int, trigger_id: int) -> int:
        self._populate_pipeline_if_exists(pipeline_id)

        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested number of samples from pipeline {pipeline_id} which does not exist!")

        return self._selectors[pipeline_id].get_number_of_samples(trigger_id)

    def get_status_bar_scale(self, pipeline_id: int) -> int:
        self._populate_pipeline_if_exists(pipeline_id)

        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested status bar scale from pipeline {pipeline_id} which does not exist!")

        return self._selectors[pipeline_id].get_status_bar_scale()

    def get_number_of_partitions(self, pipeline_id: int, trigger_id: int) -> int:
        self._populate_pipeline_if_exists(pipeline_id)

        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested number of partitions from pipeline {pipeline_id} which does not exist!")

        return self._selectors[pipeline_id].get_number_of_partitions(trigger_id)

    def get_available_labels(self, pipeline_id: int) -> list[int]:
        self._populate_pipeline_if_exists(pipeline_id)

        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested available labels from pipeline {pipeline_id} which does not exist!")

        return self._selectors[pipeline_id].get_available_labels()

    def uses_weights(self, pipeline_id: int) -> bool:
        self._populate_pipeline_if_exists(pipeline_id)

        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested whether the pipeline {pipeline_id} uses weights but it does not exist!")

        return self._selectors[pipeline_id].uses_weights()

    def _instantiate_strategy(
        self, selection_strategy: SelectionStrategyModel, pipeline_id: int
    ) -> AbstractSelectionStrategy:
        strategy_name = selection_strategy.name
        strategy_module = dynamic_module_import("modyn.selector.internal.selector_strategies")
        if not hasattr(strategy_module, strategy_name):
            raise NotImplementedError(f"Strategy {strategy_name} not available!")

        strategy_handler = getattr(strategy_module, strategy_name)

        return strategy_handler(selection_strategy, self._modyn_config, pipeline_id)

    def get_selection_strategy_remote(self, pipeline_id: int) -> tuple[bool, str, dict]:
        self._populate_pipeline_if_exists(pipeline_id)

        if pipeline_id not in self._selectors:
            raise ValueError(f"Requested selection strategy for pipeline {pipeline_id} which does not exist!")

        return self._selectors[pipeline_id].get_selection_strategy_remote()

    def cleanup_trigger_samples(self) -> None:
        if (
            "cleanup_storage_directories_after_shutdown" in self._modyn_config["selector"]
            and "trigger_sample_directory" in self._modyn_config["selector"]
        ):
            shutil.rmtree(self._modyn_config["selector"]["trigger_sample_directory"])
            Path(self._modyn_config["selector"]["trigger_sample_directory"]).mkdir(parents=True, exist_ok=True)
            logger.info("Deleted the trigger sample directory.")

    def cleanup_local_storage(self) -> None:
        if (
            "cleanup_storage_directories_after_shutdown" in self._modyn_config["selector"]
            and "local_storage_directory" in self._modyn_config["selector"]
        ):
            shutil.rmtree(self._modyn_config["selector"]["local_storage_directory"])
            Path(self._modyn_config["selector"]["local_storage_directory"]).mkdir(parents=True, exist_ok=True)
            logger.info("Deleted the local storage directory.")
