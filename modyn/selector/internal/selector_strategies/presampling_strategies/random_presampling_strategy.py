from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies.abstract_presampling_strategy import (
    AbstractPresamplingStrategy,
)
from sqlalchemy import Select, asc, func, select


class RandomPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(
        self,
        config: dict,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
        tail_triggers: int,
        has_limit: bool,
        training_set_size_limit: int,
    ):
        super().__init__(
            config, modyn_config, pipeline_id, maximum_keys_in_memory, tail_triggers, has_limit, training_set_size_limit
        )

        if "presampling_ratio" not in config:
            raise ValueError(
                "Please specify the presampling ratio. If you want to avoid presampling, set presampling_ratio to 100"
            )
        self.presampling_ratio = config["presampling_ratio"]

        if not (0 < self.presampling_ratio < 100) or not isinstance(self.presampling_ratio, int):
            raise ValueError("Presampling ratio must be an integer in range (0,100)")

    def get_presampling_target_size(self, next_trigger_id: int) -> int:
        dataset_size = self._get_dataset_size(next_trigger_id)
        target_presampling = (dataset_size * self.presampling_ratio) // 100
        return target_presampling

    def _get_dataset_size(self, next_trigger_id: int) -> int:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            return (
                database.session.query(SelectorStateMetadata.sample_key)
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id,
                    SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - self._tail_triggers
                    if self._tail_triggers is not None
                    else True,
                )
                .count()
            )

    def get_query_stmt(self, next_trigger_id: int) -> Select:
        presampling_target_size = self.get_presampling_target_size(next_trigger_id)

        if self._has_limit:
            target_size = min(self._training_set_size_limit, presampling_target_size)
        else:
            target_size = presampling_target_size

        subq = (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == self._pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - self._tail_triggers
                if self._tail_triggers is not None
                else True,
            )
            .order_by(func.random())  # pylint: disable=E1102
            .limit(target_size)
        )

        stmt = (
            select(SelectorStateMetadata.sample_key)
            .execution_options(yield_per=self._maximum_keys_in_memory)
            .filter(
                SelectorStateMetadata.pipeline_id == self._pipeline_id,
                SelectorStateMetadata.sample_key.in_(subq),
            )
            .order_by(asc(SelectorStateMetadata.timestamp))
        )

        return stmt
