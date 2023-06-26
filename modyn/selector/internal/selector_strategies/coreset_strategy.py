import random
from typing import Iterable

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies import (
    AbstractDownsamplingStrategy,
    EmptyDownsamplingStrategy,
)
from modyn.selector.internal.selector_strategies.presampling_strategies import (
    AbstractPresamplingStrategy,
    EmptyPresamplingStrategy,
)
from modyn.utils import dynamic_module_import


class CoresetStrategy(AbstractSelectionStrategy):
    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        self.presampling_strategy: AbstractPresamplingStrategy = self._instantiate_presampler(
            config, modyn_config, pipeline_id, maximum_keys_in_memory
        )

        self.downsampling_strategy: AbstractDownsamplingStrategy = self._instantiate_downsampler(config)

        if isinstance(self.presampling_strategy, EmptyPresamplingStrategy) and isinstance(
            self.downsampling_strategy, EmptyDownsamplingStrategy
        ):
            raise ValueError(
                "Using a Coreset Method without presampling and downsampling is useless. "
                "You can use NewDataStrategy instead. "
                "To specify the presampling method add 'presampling_strategy to the pipeline. "
                "You can use 'downsampling_strategies' to specify the downsampling method."
            )

    def _instantiate_presampler(
        self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int
    ) -> AbstractPresamplingStrategy:
        presampling_strategy_module = dynamic_module_import(
            "modyn.selector.internal.selector_strategies.presampling_strategies"
        )
        if "presampling_strategy" not in config:
            presampling_strategy = "EmptyPresamplingStrategy"
        else:
            presampling_strategy = config["presampling_strategy"]

        # for simplicity, you can just specify the short name (without PresamplingStrategy)
        if not hasattr(presampling_strategy_module, presampling_strategy):
            long_name = f"{presampling_strategy}PresamplingStrategy"
            if not hasattr(presampling_strategy_module, long_name):
                raise ValueError("Requested presampling strategy does not exist")
            presampling_strategy = long_name
        presampling_class = getattr(presampling_strategy_module, presampling_strategy)
        return presampling_class(
            config,
            modyn_config,
            pipeline_id,
            maximum_keys_in_memory,
        )

    def _instantiate_downsampler(
        self,
        config: dict,
    ) -> AbstractDownsamplingStrategy:
        downsampling_strategy_module = dynamic_module_import(
            "modyn.selector.internal.selector_strategies.downsampling_strategies"
        )
        if "downsampling_strategy" not in config:
            downsampling_strategy = "EmptyDownsamplingStrategy"
        else:
            downsampling_strategy = config["downsampling_strategy"]

        # for simplicity, you can just specify the short name (without DownsamplingStrategy)
        if not hasattr(downsampling_strategy_module, downsampling_strategy):
            long_name = f"{downsampling_strategy}DownsamplingStrategy"
            if not hasattr(downsampling_strategy_module, long_name):
                raise ValueError("Requested presampling strategy does not exist")
            downsampling_strategy = long_name

        downsampling_class = getattr(downsampling_strategy_module, downsampling_strategy)
        return downsampling_class(
            config,
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
            if self.presampling_strategy.requires_trigger_dataset_size():
                target_size = self._get_dataset_size()
            else:
                target_size = None
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
