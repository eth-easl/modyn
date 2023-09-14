import random
from typing import Iterable

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies import (
    DownsamplingScheduler,
    instantiate_scheduler,
)
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from modyn.selector.internal.selector_strategies.presampling_strategies.utils import instantiate_presampler


class CoresetStrategy(AbstractSelectionStrategy):
    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        # Every coreset method has a presampling strategy to select datapoints to train on
        self.presampling_strategy: AbstractPresamplingStrategy = instantiate_presampler(
            config, modyn_config, pipeline_id
        )
        # and a downsampler scheduler to downsample the data at the trainer server. The scheduler might just be a single
        # strategy.
        self.downsampling_scheduler: DownsamplingScheduler = instantiate_scheduler(config, maximum_keys_in_memory)

    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> dict[str, object]:
        assert len(keys) == len(timestamps)
        assert len(timestamps) == len(labels)

        swt = Stopwatch()
        swt.start("persist_samples")
        persist_log = self._persist_samples(keys, timestamps, labels)
        return {"total_persist_time": swt.stop(), "persist_log": persist_log}

    def trigger(self) -> tuple[int, int, int, dict[str, object]]:
        trigger_id, total_keys_in_trigger, num_partitions, log = super().trigger()
        self.downsampling_scheduler.inform_next_trigger(self._next_trigger_id)
        return trigger_id, total_keys_in_trigger, num_partitions, log

    def _on_trigger(self) -> Iterable[tuple[list[tuple[int, float]], dict[str, object]]]:
        for samples in self._get_data():
            random.shuffle(samples)
            # Add logging here when required.
            yield [(sample, 1.0) for sample in samples], {}

    def _get_data(self) -> Iterable[list[int]]:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            trigger_dataset_size = None
            if self.presampling_strategy.requires_trigger_dataset_size:
                trigger_dataset_size = self._get_trigger_dataset_size()

            stmt = self.presampling_strategy.get_presampling_query(
                next_trigger_id=self._next_trigger_id,
                tail_triggers=self.tail_triggers,
                limit=self.training_set_size_limit if self.has_limit else None,
                trigger_dataset_size=trigger_dataset_size,
            ).execution_options(yield_per=self._maximum_keys_in_memory)

            for chunk in database.session.execute(stmt).partitions():
                if len(chunk) > 0:
                    yield [res[0] for res in chunk]
                else:
                    yield []

    def _reset_state(self) -> None:
        pass  # As we currently hold everything in database (#116), this currently is a noop.

    def _get_trigger_dataset_size(self) -> int:
        # Count the number of samples that might be sampled during the next trigger. Typically used to compute the
        # target size for presampling_strategies (target_size = trigger_dataset_size * ratio)
        with MetadataDatabaseConnection(self._modyn_config) as database:
            return (
                database.session.query(SelectorStateMetadata.sample_key)
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id,
                    SelectorStateMetadata.seen_in_trigger_id >= self._next_trigger_id - self.tail_triggers
                    if self.tail_triggers is not None
                    else True,
                )
                .count()
            )

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
