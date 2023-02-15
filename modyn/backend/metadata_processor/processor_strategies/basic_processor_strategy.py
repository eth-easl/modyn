import json

from modyn.backend.metadata_database.models import SampleTrainingMetadata, TriggerTrainingMetadata
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

    @abstractmethod
    def process_trigger_metadata(self,
        trigger_metadata: PerTriggerMetadata
        ) -> dict:
        if trigger_metadata and trigger_metadata.loss:
            return {
                "loss": trigger_metadata.loss
            }
        return None

    @abstractmethod
    def process_sample_metadata(self,
        sample_metadata: Iterable[PerSampleMetadata]
        ) -> list[dict]:
        if sample_metadata:
            processed_metadata = [
                {"sample_id": metadata.sample_id, "loss": metadata.loss}
                for metadata in sample_metadata
            ]
            return processed_metadata
        return None
