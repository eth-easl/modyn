from abc import ABC, abstractmethod
from typing import Iterable, Optional

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SampleTrainingMetadata, TriggerTrainingMetadata

# pylint: disable-next=no-name-in-module
from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (  # noqa: E402, E501
    PerSampleMetadata,
    PerTriggerMetadata,
)
from modyn.backend.metadata_processor.processor_strategies.processor_strategy_type import ProcessorStrategyType


class AbstractProcessorStrategy(ABC):
    """This class is the base class for Metadata Processors. In order to extend
    this class to perform custom processing, implement process_trigger_metadata
    and process_sample_metadata
    """

    processor_strategy_type: ProcessorStrategyType = None

    @abstractmethod
    def process_trigger_metadata(self, trigger_metadata: PerTriggerMetadata) -> Optional[dict]:
        raise NotImplementedError()

    @abstractmethod
    def process_sample_metadata(self, sample_metadata: Iterable[PerSampleMetadata]) -> Optional[list[dict]]:
        raise NotImplementedError()
