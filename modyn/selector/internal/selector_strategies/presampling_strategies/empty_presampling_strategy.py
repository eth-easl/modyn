from typing import Optional

from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from sqlalchemy import Select, asc, select


class EmptyPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        if "presampling_ratio" in config and config["presampling_ratio"] != 100:
            raise ValueError("Using EmptyPresamplingStrategy, the presampling_ratio is implicitly 100%")

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: Optional[int],
        limit: Optional[int],
        trigger_dataset_size: Optional[int],
    ) -> Select:
        stmt = (
            select(SelectorStateMetadata.sample_key)
            # Enables batching of results in chunks.
            # See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
            .execution_options(yield_per=self.maximum_keys_in_memory)
            .filter(
                SelectorStateMetadata.pipeline_id == self.pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - tail_triggers
                if tail_triggers is not None
                else True,
            )
            .order_by(asc(SelectorStateMetadata.timestamp))
        )

        if limit is not None:
            stmt = stmt.limit(limit)

        return stmt

    def requires_trigger_dataset_size(
        self,
    ) -> bool:
        return False
