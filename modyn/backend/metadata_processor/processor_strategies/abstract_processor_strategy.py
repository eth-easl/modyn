from abc import ABC, abstractmethod

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SampleTrainingMetadata, TriggerTrainingMetadata
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
        processed_trigger_metadata = self.process_trigger_metadata(trigger_metadata)
        processed_sample_metadata = self.process_sample_metadata(sample_metadata)

        with MetadataDatabaseConnection(self.config) as database:
            if processed_trigger_metadata:
                database.session.add(
                    TriggerTrainingMetadata(
                        trigger_id=trigger_id,
                        pipeline_id=pipeline_id,
                        overall_loss=processed_trigger_metadata.get("loss", None),
                        time_to_train=processed_trigger_metadata.get("time", None)))
                database.session.commit()
            if processed_sample_metadata:
                database.session.add_all(
                    [
                        SampleTrainingMetadata(
                            pipeline_id=pipeline_id,
                            trigger_id=trigger_id,
                            sample_key=metadata["sample_id"],
                            loss=metadata.get("loss", None),
                            gradient=metadata.get("gradient", None))
                        for metadata in processed_sample_metadata
                    ]
                )
                database.session.commit()

    @abstractmethod
    def process_trigger_metadata(self,
        trigger_metadata: PerTriggerMetadata
        ) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def process_sample_metadata(self,
        sample_metadata: Iterable[PerSampleMetadata]
        ) -> list[dict]:
        raise NotImplementedError()
