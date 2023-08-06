from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractBalancedPresamplingStrategy


class LabelBalancedPresamplingStrategy(AbstractBalancedPresamplingStrategy):
    def __init__(self, presampling_config: dict, modyn_config: dict, pipeline_id: int):
        super().__init__(presampling_config, modyn_config, pipeline_id, SelectorStateMetadata.label)
