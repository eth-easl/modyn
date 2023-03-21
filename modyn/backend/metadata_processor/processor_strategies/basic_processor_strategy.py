from typing import Iterable, Optional

# pylint: disable-next=no-name-in-module
from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (  # noqa: E402, E501
    PerSampleMetadata,
    PerTriggerMetadata,
)
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy
from modyn.backend.metadata_processor.processor_strategies.processor_strategy_type import ProcessorStrategyType


class BasicProcessorStrategy(AbstractProcessorStrategy):
    """This class represents a basic Metadata Processor strategy that takes the
    values sent by the Collector and stores them in the Database
    """

    def __init__(self, modyn_config: dict, pipeline_id: int):
        super().__init__(modyn_config, pipeline_id)
        self.processor_strategy_type = ProcessorStrategyType.BasicProcessorStrategy

    def process_trigger_metadata(self, trigger_metadata: PerTriggerMetadata) -> Optional[dict]:
        if trigger_metadata and trigger_metadata.loss:
            return {"loss": trigger_metadata.loss}
        return None

    def process_sample_metadata(self, sample_metadata: Iterable[PerSampleMetadata]) -> Optional[list[dict]]:
        if sample_metadata:
            processed_metadata = [
                {"sample_id": metadata.sample_id, "loss": metadata.loss}
                for metadata in sample_metadata
                if metadata.loss is not None
            ]
            return processed_metadata
        return None
