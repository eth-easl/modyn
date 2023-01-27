"""GRPC Servicer for Metadata Processor"""

import logging
from typing import Iterable

import grpc

# pylint: disable-next=no-name-in-module
from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (
    TrainingMetadataRequest,
    TrainingMetadataResponse,
)
from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2_grpc import (
    MetadataProcessorServicer,
)
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy

logger = logging.getLogger(__name__)


class MetadataProcessorGRPCServicer(MetadataProcessorServicer):
    """GRPC Servicer for the MetadataProcessor module."""

    def __init__(self, strategy: AbstractProcessorStrategy) -> None:
        super().__init__()
        self.processor_strategy = strategy

    def ProcessTrainingMetadata(
        self, request_iterator: Iterable[TrainingMetadataRequest], context: grpc.ServicerContext
    ) -> TrainingMetadataResponse:
        """Process training metadata.

        Args:
            request_iterator (Iterable[TrainingMetadataRequest]): Requests
                containing the training ID and the metadata
            context (grpc.ServicerContext): Context of the request.

        Returns:
            response (TrainingMetadataResponse): Empty response, to confirm.
        """
        for request in request_iterator:
            logger.info(f"Processing training metadata for pipeline ID {request.pipeline_id} and trigger ID {request.trigger_id}")
            self.processor_strategy.process_training_metadata(
                request.pipeline_id, request.trigger_id, request.trigger_metadata, request.sample_metadata)

        return TrainingMetadataResponse()
