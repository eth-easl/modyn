import random
from typing import Iterable

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.utils import dynamic_module_import


class GeneralPresamplingStrategy(AbstractSelectionStrategy):
    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        presampling_strategy_module = dynamic_module_import(
            "modyn.selector.internal.selector_strategies.presampling_strategies"
        )

        if "presampling_strategy" not in config:
            presampling_strategy = "AllDataPresamplingStrategy"
        else:
            presampling_strategy = config["presampling_strategy"]

        if not hasattr(presampling_strategy_module, presampling_strategy):
            raise ValueError("Requested presampling strategy does not exist")
        presampling_class = getattr(presampling_strategy_module, presampling_strategy)
        self._presampling_strategy = presampling_class(
            config,
            modyn_config,
            pipeline_id,
            maximum_keys_in_memory,
            self.tail_triggers,
            self.has_limit,
            self.training_set_size_limit,
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
            stmt = self._presampling_strategy.get_query_stmt(self._next_trigger_id)

            for chunk in database.session.execute(stmt).partitions():
                if len(chunk) > 0:
                    yield [res[0] for res in chunk]
                else:
                    yield []

    def _reset_state(self) -> None:
        pass  # As we currently hold everything in database (#116), this currently is a noop.
