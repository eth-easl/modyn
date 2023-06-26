from typing import Optional

from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies.abstract_presampling_strategy import (
    AbstractPresamplingStrategy,
)
from sqlalchemy import Select, asc, func, select


class RandomPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        if "presampling_ratio" not in config:
            raise ValueError(
                "Please specify the presampling ratio. If you want to avoid presampling, set presampling_ratio to 100"
            )
        self.presampling_ratio = config["presampling_ratio"]

        if not (0 < self.presampling_ratio < 100) or not isinstance(self.presampling_ratio, int):
            raise ValueError("Presampling ratio must be an integer in range (0,100)")

    def get_presampling_target_size(self, trigger_dataset_size: int) -> int:
        assert trigger_dataset_size >= 0
        target_presampling = int(trigger_dataset_size * self.presampling_ratio / 100)
        return target_presampling

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: Optional[int],
        limit: Optional[int],
        trigger_dataset_size: Optional[int],
    ) -> Select:
        # TODO(#224) write an efficient query using TABLESAMPLE
        assert trigger_dataset_size is not None
        assert trigger_dataset_size >= 0

        presampling_target_size = self.get_presampling_target_size(trigger_dataset_size)

        if limit is not None:
            assert limit >= 0
            target_size = min(limit, presampling_target_size)
        else:
            target_size = presampling_target_size

        subq = (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == self.pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - tail_triggers
                if tail_triggers is not None
                else True,
            )
            .order_by(func.random())  # pylint: disable=E1102
            .limit(target_size)
        )

        stmt = (
            select(SelectorStateMetadata.sample_key)
            .execution_options(yield_per=self.maximum_keys_in_memory)
            .filter(
                SelectorStateMetadata.pipeline_id == self.pipeline_id,
                SelectorStateMetadata.sample_key.in_(subq),
            )
            .order_by(asc(SelectorStateMetadata.timestamp))
        )

        return stmt

    def requires_trigger_dataset_size(
        self,
    ) -> bool:
        return True
