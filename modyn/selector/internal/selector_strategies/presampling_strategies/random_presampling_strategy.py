from typing import Optional

from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies.abstract_presampling_strategy import (
    AbstractPresamplingStrategy,
)
from sqlalchemy import Select, asc, func, select


class RandomPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(self, presampling_config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(presampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self.requires_trigger_dataset_size = True

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: Optional[int],
        limit: Optional[int],
        trigger_dataset_size: Optional[int],
        requires_samples_ordered_by_label: bool,
    ) -> Select:
        # TODO(#224) write an efficient query using TABLESAMPLE
        assert trigger_dataset_size is not None
        assert trigger_dataset_size >= 0

        target_size = self.get_target_size(trigger_dataset_size, limit)

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
        )

        if requires_samples_ordered_by_label:
            stmt = stmt.order_by(SelectorStateMetadata.label)
        else:
            stmt = stmt.order_by(asc(SelectorStateMetadata.timestamp))

        return stmt
