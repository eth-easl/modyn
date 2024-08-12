"""GRPC Servicer for Metadata Processor."""

import logging

import grpc

# pylint: disable-next=no-name-in-module
from modyn.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (
    PipelineResponse,
    RegisterPipelineRequest,
    TrainingMetadataRequest,
    TrainingMetadataResponse,
)
from modyn.metadata_processor.internal.grpc.generated.metadata_processor_pb2_grpc import MetadataProcessorServicer
from modyn.metadata_processor.internal.metadata_processor_manager import MetadataProcessorManager

logger = logging.getLogger(__name__)


class MetadataProcessorGRPCServicer(MetadataProcessorServicer):
    """GRPC Servicer for the MetadataProcessor module."""

    def __init__(self, processor_manager: MetadataProcessorManager) -> None:
        super().__init__()
        self.processor_manager = processor_manager

    def register_pipeline(self, request: RegisterPipelineRequest, context: grpc.ServicerContext) -> PipelineResponse:
        logger.info(f"Registering pipeline with request - {str(request)}")
        self.processor_manager.register_pipeline(request.pipeline_id, request.processor_type)
        return PipelineResponse()

    def process_training_metadata(
        self, request: TrainingMetadataRequest, context: grpc.ServicerContext
    ) -> TrainingMetadataResponse:
        # TODO(#210): This needs to be done in a streaming fashion to avoid scalability issues
        logger.info(
            f"Processing training metadata for pipeline ID {request.pipeline_id}"
            f" and trigger ID {request.trigger_id}"
        )
        self.processor_manager.process_training_metadata(
            request.pipeline_id, request.trigger_id, request.trigger_metadata, request.sample_metadata
        )

        return TrainingMetadataResponse()
