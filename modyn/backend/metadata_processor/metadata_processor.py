"""Metadata Processor module.

The Metadata Processor module contains all classes and functions related to the
post-training processing of metadata collected by the GPU Node.
"""

import logging

from modyn.backend.metadata_processor.internal.grpc.grpc_server import GRPCServer
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy
from modyn.backend.metadata_processor.processor_strategies.processor_strategy_type import ProcessorStrategyType
from modyn.utils import dynamic_module_import

logger = logging.getLogger(__name__)


class MetadataProcessor:
    def __init__(self, strategy: AbstractProcessorStrategy, pipeline_id: int) -> None:
        self.strategy = strategy
        self.pipeline_id = pipeline_id

    def process_training_metadata(
        self,
        pipeline_id: int,
        trigger_id: int,
        trigger_metadata: PerTriggerMetadata,
        sample_metadata: Iterable[PerSampleMetadata],
    ) -> None:
        processed_trigger_metadata = self.strategy.process_trigger_metadata(trigger_metadata)
        processed_sample_metadata = self.strategy.process_sample_metadata(sample_metadata)
        self.persist_metadata(processed_trigger_metadata, processed_sample_metadata)
        
    def persist_metadata(
        self,
        processed_trigger_metadata: Optional[dict],
        processed_sample_metadata: Optional[list[dict]]
    ) -> None:
        with MetadataDatabaseConnection(self.config) as database:
            if processed_trigger_metadata is not None:
                database.session.add(
                    TriggerTrainingMetadata(
                        trigger_id=trigger_id,
                        pipeline_id=pipeline_id,
                        overall_loss=processed_trigger_metadata.get("loss", None),
                        time_to_train=processed_trigger_metadata.get("time", None),
                    )
                )
                database.session.commit()
            if processed_sample_metadata is not None:
                database.session.add_all(
                    [
                        SampleTrainingMetadata(
                            pipeline_id=pipeline_id,
                            trigger_id=trigger_id,
                            sample_key=metadata["sample_id"],
                            loss=metadata.get("loss", None),
                            gradient=metadata.get("gradient", None),
                        )
                        for metadata in processed_sample_metadata
                    ]
                )
                database.session.commit()