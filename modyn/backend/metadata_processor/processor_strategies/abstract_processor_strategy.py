from abc import ABC, abstractmethod

from modyn.backend.metadata_processor.internal.grpc.grpc_handler import GRPCHandler
from modyn.backend.metadata_processor.processor_strategies.processor_strategy_type import ProcessorStrategyType
from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (
    PerTriggerMetadata, PerSampleMetadata
)


class AbstractProcessorStrategy(ABC):
    """This class is the base class for Metadata Processors. In order to extend
    this class to perform custom processing, implement process_metadata
    """

    processor_strategy_type: ProcessorStrategyType = None

    def __init__(self, modyn_config: dict):
        self.config = modyn_config
        self.grpc = GRPCHandler(modyn_config)

    def process_training_metadata(
        self, pipeline_id: int, trigger_id: int,
        trigger_metadata: PerTriggerMetadata,
        sample_metadata: Iterable[PerSampleMetadata]
        ) -> None:
        """
        Process the metadata and send it to the Metadata Database.

        Args:
            training_id (int): The training ID.
            data (str): Serialized training metadata.
        """
        data = self.process_metadata(trigger_metadata, sample_metadata)
        self.grpc.set_metadata(pipeline_id, trigger_id, data)

    @abstractmethod
    def process_metadata(self,
        trigger_metadata: PerTriggerMetadata,
        sample_metadata: Iterable[PerSampleMetadata]
        ) -> dict:
        raise NotImplementedError()
