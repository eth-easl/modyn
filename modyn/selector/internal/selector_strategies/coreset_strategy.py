import logging
import random
from collections.abc import Iterable

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.config.schema.pipeline import CoresetStrategyConfig
from modyn.config.schema.pipeline.sampling.config import PresamplingConfig
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies import (
    DownsamplingScheduler,
    instantiate_scheduler,
)
from modyn.selector.internal.selector_strategies.presampling_strategies import (
    AbstractPresamplingStrategy,
    NoPresamplingStrategy,
)
from modyn.selector.internal.selector_strategies.presampling_strategies.utils import instantiate_presampler
from modyn.selector.internal.selector_strategies.utils import get_trigger_dataset_size
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.selector.internal.storage_backend.database import DatabaseStorageBackend

logger = logging.getLogger(__name__)


class CoresetStrategy(AbstractSelectionStrategy):
    def __init__(self, config: CoresetStrategyConfig, modyn_config: dict, pipeline_id: int):
        super().__init__(config, modyn_config, pipeline_id)
        # a downsampler scheduler to downsample the data at the trainer server. The scheduler might just be a single
        # strategy.
        self.warmup_triggers = config.warmup_triggers
        self.downsampling_scheduler: DownsamplingScheduler = instantiate_scheduler(config, modyn_config, pipeline_id)
        # Every coreset method has a presampling strategy to select datapoints to train on
        self.presampling_strategy: AbstractPresamplingStrategy = instantiate_presampler(
            config, modyn_config, pipeline_id, self._storage_backend
        )

        self.warmup_presampler = NoPresamplingStrategy(
            PresamplingConfig(strategy="No", ratio=100), modyn_config, pipeline_id, self._storage_backend
        )

    def _init_storage_backend(self) -> AbstractStorageBackend:
        if self._config.storage_backend == "local":
            # TODO(#324): Support local backend on CoresetStrategy
            raise NotImplementedError("The CoresetStrategy currently does not support the local backend.")

        if self._config.storage_backend == "database":
            storage_backend = DatabaseStorageBackend(
                self._pipeline_id, self._modyn_config, self._maximum_keys_in_memory
            )
        else:
            raise NotImplementedError(f'Unknown storage backend "{self._config.storage_backend}". Supported: database')
        return storage_backend

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

        downsampler_log = self.downsampling_scheduler.inform_next_trigger(self._next_trigger_id, self._storage_backend)
        trigger_id, total_keys_in_trigger, num_partitions, tss_log = super().trigger()
        log: dict[str, object] = {"downsampler_log": downsampler_log, "trigger_sample_storage_log": tss_log}
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

        next_trigger_is_warmup = self._next_trigger_id + 1 <= self.warmup_triggers
        presampling_strategy = self.warmup_presampler if next_trigger_is_warmup else self.presampling_strategy
        trigger_dataset_size = None
        if presampling_strategy.requires_trigger_dataset_size:
            trigger_dataset_size = self._get_trigger_dataset_size()

        stmt = presampling_strategy.get_presampling_query(
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
        return get_trigger_dataset_size(
            self._storage_backend, self._pipeline_id, self._next_trigger_id, self.tail_triggers
        )

    def get_available_labels(self) -> list[int]:
        return self._storage_backend.get_available_labels(self._next_trigger_id, tail_triggers=self.tail_triggers)

    @property
    def downsampling_strategy(self) -> str:
        # The strategy we want to query is not the _next_trigger_id's strategy, but the current trigger's strategy
        # whose TSS is already computed. Therefore here we shouldn't + 1 to _next_trigger_id
        prev_trigger_is_warmup = self._next_trigger_id <= self.warmup_triggers
        return self.downsampling_scheduler.downsampling_strategy if not prev_trigger_is_warmup else ""

    @property
    def downsampling_params(self) -> dict:
        # The strategy we want to query is not the _next_trigger_id's strategy, but the current trigger's strategy
        # whose TSS is already computed. Therefore here we shouldn't + 1 to _next_trigger_id
        prev_trigger_is_warmup = self._next_trigger_id <= self.warmup_triggers
        return self.downsampling_scheduler.downsampling_params if not prev_trigger_is_warmup else {}

    @property
    def requires_remote_computation(self) -> bool:
        # The strategy we want to query is not the _next_trigger_id's strategy, but the current trigger's strategy
        # whose TSS is already computed. Therefore here we shouldn't + 1 to _next_trigger_id
        prev_trigger_is_warmup = self._next_trigger_id <= self.warmup_triggers
        return self.downsampling_scheduler.requires_remote_computation if not prev_trigger_is_warmup else False

    @property
    def training_status_bar_scale(self) -> int:
        return self.downsampling_scheduler.training_status_bar_scale
