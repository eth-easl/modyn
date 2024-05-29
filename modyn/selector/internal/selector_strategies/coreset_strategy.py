import logging
import random
from typing import Any, Iterable

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.config import CoresetSelectionConfig
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies import (
    DownsamplingScheduler,
    instantiate_scheduler,
)
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from modyn.selector.internal.selector_strategies.presampling_strategies.utils import instantiate_presampler
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.selector.internal.storage_backend.database import DatabaseStorageBackend
from sqlalchemy.orm.session import Session

logger = logging.getLogger(__name__)


class CoresetStrategy(AbstractSelectionStrategy):
    def __init__(self, config: CoresetSelectionConfig, modyn_config: dict, pipeline_id: int):
        super().__init__(config, modyn_config, pipeline_id)

        # and a downsampler scheduler to downsample the data at the trainer server. The scheduler might just be a single
        # strategy.
        self.downsampling_scheduler: DownsamplingScheduler = instantiate_scheduler(config, modyn_config, pipeline_id)
        self._storage_backend: AbstractStorageBackend

        if config.storage_backend == "local":
            # TODO(#324): Support local backend on CoresetStrategy
            raise NotImplementedError("The CoresetStrategy currently does not support the local backend.")

        if config.storage_backend == "database":
            self._storage_backend = DatabaseStorageBackend(
                self._pipeline_id, self._modyn_config, self._maximum_keys_in_memory
            )
        else:
            raise NotImplementedError(f'Unknown storage backend "{config.storage_backend}". Supported: database')

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
        # Upon entering this method, self._next_trigger_id is the trigger id for the current trigger
        # whose training set is to be computed. After calling super().trigger() self._next_trigger_id is incremented.

        # Therefore we need to update the downsampler before calling super().trigger().
        # We should only update the downsampler to the current trigger not to the future one referred by
        # self._next_trigger_id after the call to super().trigger() because later the remote downsampler on trainer
        # server needs to fetch configuration for the current trigger
        self.downsampling_scheduler.inform_next_trigger(self._next_trigger_id, self._storage_backend)
        trigger_id, total_keys_in_trigger, num_partitions, log = super().trigger()
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
            trigger_dataset_size = self._get_trigger_dataset_size()

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

    def _get_trigger_dataset_size(self) -> int:
        # Count the number of samples that might be sampled during the next trigger. Typically used to compute the
        # target size for presampling_strategies (target_size = trigger_dataset_size * ratio)
        assert isinstance(
            self._storage_backend, DatabaseStorageBackend
        ), "CoresetStrategy currently only supports DatabaseBackend"

        def _session_callback(session: Session) -> Any:
            return (
                session.query(SelectorStateMetadata.sample_key)
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id,
                    (
                        SelectorStateMetadata.seen_in_trigger_id >= self._next_trigger_id - self.tail_triggers
                        if self.tail_triggers is not None
                        else True
                    ),
                )
                .count()
            )

        return self._storage_backend._execute_on_session(_session_callback)

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
