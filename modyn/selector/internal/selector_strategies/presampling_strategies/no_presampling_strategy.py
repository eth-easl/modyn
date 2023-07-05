from typing import Optional

from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from sqlalchemy import Select, asc, select


class NoPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(self, presampling_config: dict, modyn_config: dict, pipeline_id: int):
        if "ratio" in presampling_config and presampling_config["ratio"] != 100:
            raise ValueError("Using NoPresamplingStrategy, the presampling_ratio is implicitly 100%")
        presampling_config["ratio"] = 100
        super().__init__(presampling_config, modyn_config, pipeline_id)

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: Optional[int],
        limit: Optional[int],
        trigger_dataset_size: Optional[int],
        requires_samples_ordered_by_label: bool,
    ) -> Select:
        stmt = select(SelectorStateMetadata.sample_key).filter(
            SelectorStateMetadata.pipeline_id == self.pipeline_id,
            SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - tail_triggers
            if tail_triggers is not None
            else True,
        )

        if requires_samples_ordered_by_label:
            stmt = stmt.order_by(SelectorStateMetadata.label)
        else:
            stmt = stmt.order_by(asc(SelectorStateMetadata.timestamp))

        if limit is not None:
            stmt = stmt.limit(limit)

        return stmt
