import logging
import random
from typing import Iterable

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies import (
    DownsamplingScheduler,
    instantiate_scheduler,
)
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from modyn.selector.internal.selector_strategies.presampling_strategies.utils import instantiate_presampler
from modyn.selector.internal.selector_strategies.utils import get_trigger_dataset_size
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.selector.internal.storage_backend.database import DatabaseStorageBackend

logger = logging.getLogger(__name__)


class CoresetStrategy(AbstractSelectionStrategy):
    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        # and a downsampler scheduler to downsample the data at the trainer server. The scheduler might just be a single
        # strategy.
        self.downsampling_scheduler: DownsamplingScheduler = instantiate_scheduler(
            config, modyn_config, pipeline_id, maximum_keys_in_memory
        )
        self._storage_backend: AbstractStorageBackend
        if "storage_backend" in config:
            if config["storage_backend"] == "local":
                # TODO(#324): Support local backend on CoresetStrategy
                raise NotImplementedError("The CoresetStrategy currently does not support the local backend.")

            if config["storage_backend"] == "database":
                self._storage_backend = DatabaseStorageBackend(
                    self._pipeline_id, self._modyn_config, self._maximum_keys_in_memory
                )
            else:
                raise NotImplementedError(
                    f"Unknown storage backend \"{config['storage_backend']}\". Supported: database"
                )
        else:
            logger.info("CoresetStrategy defaulting to database backend.")
            self._storage_backend = DatabaseStorageBackend(
                self._pipeline_id, self._modyn_config, self._maximum_keys_in_memory
            )

        # Every coreset method has a presampling strategy to select datapoints to train on
        self.presampling_strategy: AbstractPresamplingStrategy = instantiate_presampler(
            config, modyn_config, pipeline_id, self._storage_backend
        )

    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> dict[str, object]:
        assert len(keys) == len(timestamps)
        assert len(timestamps) == len(labels)

        swt = Stopwatch()
        swt.start("persist_samples")
        persist_log = self._storage_backend.persist_samples(self._next_trigger_id, keys, timestamps, labels)
        return {"total_persist_time": swt.stop(), "persist_log": persist_log}

    def trigger(self) -> tuple[int, int, int, dict[str, object]]:
        trigger_id, total_keys_in_trigger, num_partitions, log = super().trigger()
        self.downsampling_scheduler.inform_next_trigger(self._next_trigger_id, self._storage_backend)
        return trigger_id, total_keys_in_trigger, num_partitions, log

    def _on_trigger(self) -> Iterable[tuple[list[tuple[int, float]], dict[str, object]]]:
        for samples in self._get_data():
            random.shuffle(samples)
            # Add logging here when required.
            yield [(sample, 1.0) for sample in samples], {}

    def _get_data(self) -> Iterable[list[int]]:
        assert isinstance(
            self._storage_backend, DatabaseStorageBackend
        ), "CoresetStrategy currently only supports DatabaseBackend"

        trigger_dataset_size = None
        if self.presampling_strategy.requires_trigger_dataset_size:
            trigger_dataset_size = get_trigger_dataset_size(
                self._storage_backend, self._pipeline_id, self._next_trigger_id, self.tail_triggers
            )

        stmt = self.presampling_strategy.get_presampling_query(
            next_trigger_id=self._next_trigger_id,
            tail_triggers=self.tail_triggers,
            limit=self.training_set_size_limit if self.has_limit else None,
            trigger_dataset_size=trigger_dataset_size,
        )

        for samples, _ in self._storage_backend._partitioned_execute_stmt(stmt, self._maximum_keys_in_memory, None):
            yield samples

    def _reset_state(self) -> None:
        pass  # As we currently hold everything in database (#116), this currently is a noop.

    def get_available_labels(self) -> list[int]:
        return self._storage_backend.get_available_labels(self._next_trigger_id, tail_triggers=self.tail_triggers)

    @property
    def downsampling_strategy(self) -> str:
        return self.downsampling_scheduler.downsampling_strategy

    @property
    def downsampling_params(self) -> dict:
        return self.downsampling_scheduler.downsampling_params

    @property
    def requires_remote_computation(self) -> bool:
        return self.downsampling_scheduler.requires_remote_computation

    @property
    def training_status_bar_scale(self) -> int:
        return self.downsampling_scheduler.training_status_bar_scale
