from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SampleTrainingMetadata, TriggerTrainingMetadata

# pylint: disable-next=no-name-in-module
from modyn.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (  # noqa: E402, E501
    PerSampleMetadata,
    PerTriggerMetadata,
)
from modyn.metadata_processor.processor_strategies.processor_strategy_type import ProcessorStrategyType


class AbstractProcessorStrategy(ABC):
    """This class is the base class for Metadata Processors. In order to extend
    this class to perform custom processing, implement process_trigger_metadata
    and process_sample_metadata
    """

    def __init__(self, modyn_config: dict, pipeline_id: int):
        self.processor_strategy_type: ProcessorStrategyType | None = None
        self.modyn_config = modyn_config
        self.pipeline_id = pipeline_id

    def process_training_metadata(
        self,
        trigger_id: int,
        trigger_metadata: PerTriggerMetadata,
        sample_metadata: Iterable[PerSampleMetadata],
    ) -> None:
        processed_trigger_metadata = self.process_trigger_metadata(trigger_metadata)
        processed_sample_metadata = self.process_sample_metadata(sample_metadata)
        self.persist_metadata(trigger_id, processed_trigger_metadata, processed_sample_metadata)

    def persist_metadata(
        self,
        trigger_id: int,
        processed_trigger_metadata: Optional[dict],
        processed_sample_metadata: Optional[list[dict]],
    ) -> None:
        with MetadataDatabaseConnection(self.modyn_config) as database:
            if processed_trigger_metadata is not None:
                database.session.add(
                    TriggerTrainingMetadata(
                        trigger_id=trigger_id,
                        pipeline_id=self.pipeline_id,
                        overall_loss=processed_trigger_metadata.get("loss", None),
                        time_to_train=processed_trigger_metadata.get("time", None),
                    )
                )
                database.session.commit()
            if processed_sample_metadata is not None:
                database.session.bulk_insert_mappings(
                    SampleTrainingMetadata,
                    [
                        {
                            "pipeline_id": self.pipeline_id,
                            "trigger_id": trigger_id,
                            "sample_key": metadata["sample_id"],
                            "loss": metadata.get("loss", None),
                            "gradient": metadata.get("gradient", None),
                        }
                        for metadata in processed_sample_metadata
                    ],
                )
                database.session.commit()

    @abstractmethod
    def process_trigger_metadata(self, trigger_metadata: PerTriggerMetadata) -> Optional[dict]:
        raise NotImplementedError()

    @abstractmethod
    def process_sample_metadata(self, sample_metadata: Iterable[PerSampleMetadata]) -> Optional[list[dict]]:
        raise NotImplementedError()
