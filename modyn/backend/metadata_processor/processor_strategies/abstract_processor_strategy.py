from abc import ABC, abstractmethod

from modyn.backend.metadata_processor.internal.grpc.grpc_handler import GRPCHandler
from modyn.backend.metadata_processor.processor_strategies.processor_strategy_type import ProcessorStrategyType


class AbstractProcessorStrategy(ABC):
    """This class is the base class for Metadata Processors. In order to extend
    this class to perform custom processing, implement process_metadata
    """

    processor_strategy_type: ProcessorStrategyType = None

    def __init__(self, modyn_config: dict):
        self.config = modyn_config
        self.grpc = GRPCHandler(modyn_config)

    def process_post_training_metadata(self, training_id: int, serialized_data: str) -> None:
        """
        Process the metadata and send it to the Metadata Database.

        Args:
            training_id (int): The training ID.
            data (str): Serialized post training metadata.
        """
        data = self.process_metadata(training_id, serialized_data)
        self.grpc.set_metadata(training_id, data)

    @abstractmethod
    def process_metadata(self, training_id: int, data: str) -> dict:
        raise NotImplementedError()
