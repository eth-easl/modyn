import random
from typing import Iterable

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies import (
    AbstractDownsamplingStrategy,
    EmptyDownsamplingStrategy,
    instantiate_downsampler,
)
from modyn.selector.internal.selector_strategies.presampling_strategies import (
    AbstractPresamplingStrategy,
    EmptyPresamplingStrategy,
    instantiate_presampler,
)


class CoresetStrategy(AbstractSelectionStrategy):
    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        self.presampling_strategy: AbstractPresamplingStrategy = instantiate_presampler(
            config, modyn_config, pipeline_id, maximum_keys_in_memory
        )

        self.downsampling_strategy: AbstractDownsamplingStrategy = instantiate_downsampler(
            config, maximum_keys_in_memory
        )

        if isinstance(self.presampling_strategy, EmptyPresamplingStrategy) and isinstance(
            self.downsampling_strategy, EmptyDownsamplingStrategy
        ):
            raise ValueError(
                "You did not specify any presampling and downsampling strategy for the CoresetStrategy. "
                "You can use NewDataStrategy instead. "
                "To specify the presampling method add 'presampling_strategy to the pipeline. "
                "You can use 'downsampling_strategies' to specify the downsampling method."
            )

    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> None:
        assert len(keys) == len(timestamps)
        assert len(timestamps) == len(labels)

        self._persist_samples(keys, timestamps, labels)

    def _on_trigger(self) -> Iterable[list[tuple[int, float]]]:
        for samples in self._get_data():
            random.shuffle(samples)
            yield [(sample, 1.0) for sample in samples]

    def _get_data(self) -> Iterable[list[int]]:
        """Returns all sample

        Returns:
            list[str]: Keys of used samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:

            target_size = None
            if self.presampling_strategy.requires_trigger_dataset_size():
                target_size = self._get_dataset_size()

            stmt = self.presampling_strategy.get_presampling_query(
                self._next_trigger_id,
                self.tail_triggers,
                self.training_set_size_limit if self.has_limit else None,
                target_size,
            )

            for chunk in database.session.execute(stmt).partitions():
                if len(chunk) > 0:
                    yield [res[0] for res in chunk]
                else:
                    yield []

    def _reset_state(self) -> None:
        pass  # As we currently hold everything in database (#116), this currently is a noop.

    def _get_dataset_size(self) -> int:
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
        return self.downsampling_strategy.get_downsampling_strategy()

    def get_downsampling_params(self) -> dict:
        return self.downsampling_strategy.get_downsampling_params()

    def get_requires_remote_computation(self) -> bool:
        return self.downsampling_strategy.get_requires_remote_computation()

    def get_training_status_bar_scale(self) -> int:
        return self.downsampling_strategy.get_training_status_bar_scale()
