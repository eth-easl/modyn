from typing import Optional

from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractBalancedPresamplingStrategy
from sqlalchemy import Select


class TriggerBalancedPresamplingStrategy(AbstractBalancedPresamplingStrategy):
    def __init__(self, presampling_config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(presampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)

        self.balanced_column = SelectorStateMetadata.seen_in_trigger_id

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: Optional[int],
        limit: Optional[int],
        trigger_dataset_size: Optional[int],
        requires_samples_ordered_by_label: bool,
    ) -> Select:
        if tail_triggers == 0:
            raise ValueError("You cannot balance across triggers if you use reset_after_trigger")
        return super().get_presampling_query(
            next_trigger_id, tail_triggers, limit, trigger_dataset_size, requires_samples_ordered_by_label
        )
