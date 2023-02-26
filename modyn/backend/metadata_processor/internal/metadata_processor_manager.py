from typing import Iterable

from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (
    PerSampleMetadata,
    PerTriggerMetadata,
)
from modyn.backend.metadata_processor.metadata_processor import MetadataProcessor
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy
from modyn.backend.metadata_processor.processor_strategies.processor_strategy_type import ProcessorStrategyType
from modyn.utils.utils import dynamic_module_import


class MetadataProcessorManager:
    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config
        self.processors: dict[int, MetadataProcessor] = {}

    def register_pipeline(self, pipeline_id: int, processor_type: str) -> None:
        processor_strategy = self._instantiate_strategy(processor_type)
        processor = MetadataProcessor(processor_strategy, pipeline_id)
        self.processors[pipeline_id] = processor

    def process_training_metadata(
        self,
        pipeline_id: int,
        trigger_id: int,
        trigger_metadata: PerTriggerMetadata,
        sample_metadata: Iterable[PerSampleMetadata],
    ) -> None:
        if pipeline_id not in self.processors:
            raise ValueError(f"Metadata sent for processing from pipeline {pipeline_id} which does not exist!")

        self.processors[pipeline_id].process_training_metadata(trigger_id, trigger_metadata, sample_metadata)

    def _instantiate_strategy(self, processor_type: str) -> AbstractProcessorStrategy:
        strategy = ProcessorStrategyType(processor_type)
        processor_strategy_module = dynamic_module_import(
            f"modyn.backend.metadata_processor.processor_strategies.{strategy.value}"
        )
        processor_strategy = getattr(processor_strategy_module, f"{strategy.name}")
        return processor_strategy(self.modyn_config)