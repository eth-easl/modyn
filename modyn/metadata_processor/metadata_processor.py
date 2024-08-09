"""Metadata Processor module.

The Metadata Processor module contains all classes and functions related
to the post-training processing of metadata collected by the GPU Node.
"""

import logging
from collections.abc import Iterable

# pylint: disable-next=no-name-in-module
from modyn.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (  # noqa: E402, E501
    PerSampleMetadata,
    PerTriggerMetadata,
)
from modyn.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy

logger = logging.getLogger(__name__)


class MetadataProcessor:
    def __init__(self, strategy: AbstractProcessorStrategy, pipeline_id: int) -> None:
        self.strategy = strategy
        self.pipeline_id = pipeline_id

    def process_training_metadata(
        self,
        trigger_id: int,
        trigger_metadata: PerTriggerMetadata,
        sample_metadata: Iterable[PerSampleMetadata],
    ) -> None:
        self.strategy.process_training_metadata(trigger_id, trigger_metadata, sample_metadata)
