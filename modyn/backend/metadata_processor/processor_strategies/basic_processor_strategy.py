import json

from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy
from modyn.backend.metadata_processor.processor_strategies.processor_strategy_type import ProcessorStrategyType


class BasicProcessorStrategy(AbstractProcessorStrategy):
    """This class represents a basic Metadata Processor strategy that directly
    saves for each key, the metadata received along with flagging the sample
    as seen.
    """

    def __init__(self, modyn_config: dict):
        super().__init__(modyn_config)
        self.processor_strategy_type = ProcessorStrategyType.BasicProcessorStrategy

    def process_metadata(self,
        trigger_metadata: PerTriggerMetadata,
        sample_metadata: Iterable[PerSampleMetadata]
        ) -> dict:
        # TODO(): redo this based on how Metadata Database interface looks like

        return {
            "keys": [],
            "data": [],
        }
