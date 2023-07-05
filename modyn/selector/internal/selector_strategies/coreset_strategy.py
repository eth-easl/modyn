import random
from typing import Iterable

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
            config, modyn_config, pipeline_id, maximum_keys_in_memory
        )
        # and a downsampler scheduler to downsample the data at the trainer server. The scheduler might just be a single
        # strategy.
        self.downsampling_scheduler: DownsamplingScheduler = instantiate_scheduler(config, maximum_keys_in_memory)

    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> None:
        assert len(keys) == len(timestamps)
        assert len(timestamps) == len(labels)

        self._persist_samples(keys, timestamps, labels)

    def _on_trigger(self) -> Iterable[list[tuple[int, float]]]:
        # we don't want to shuffle if the downsampler needs the samples ordered by label. THe downsampler will take care
        # of shuffling the samples.
        downsampler_requires_samples_ordered_by_label = (
            self.downsampling_scheduler.get_requires_samples_ordered_by_label(next_trigger_id=self._next_trigger_id)
        )

        for samples in self._get_data():
            if not downsampler_requires_samples_ordered_by_label:
                random.shuffle(samples)
            yield [(sample, 1.0) for sample in samples]

    def _get_data(self) -> Iterable[list[int]]:

        with MetadataDatabaseConnection(self._modyn_config) as database:
            trigger_dataset_size = None
            if self.presampling_strategy.requires_trigger_dataset_size:
                trigger_dataset_size = self._get_trigger_dataset_size()

            # Some downsampling strategies require to work label by label (like CRAIG).
            # If so, we can supply samples sorted by label adding a simple ORDER BY at the end of the query
            downsampler_requires_samples_ordered_by_label = (
                self.downsampling_scheduler.get_requires_samples_ordered_by_label(self._next_trigger_id)
            )

            stmt = self.presampling_strategy.get_presampling_query(
                next_trigger_id=self._next_trigger_id,
                tail_triggers=self.tail_triggers,
                limit=self.training_set_size_limit if self.has_limit else None,
                trigger_dataset_size=trigger_dataset_size,
                requires_samples_ordered_by_label=downsampler_requires_samples_ordered_by_label,
            )

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

    def get_downsampling_strategy(self) -> str:
        return self.downsampling_scheduler.get_downsampling_strategy(self._next_trigger_id)

    def get_downsampling_params(self) -> dict:
        return self.downsampling_scheduler.get_downsampling_params(self._next_trigger_id)

    def get_requires_remote_computation(self) -> bool:
        return self.downsampling_scheduler.get_requires_remote_computation(self._next_trigger_id)

    def get_training_status_bar_scale(self) -> int:
        return self.downsampling_scheduler.get_training_status_bar_scale(self._next_trigger_id)
