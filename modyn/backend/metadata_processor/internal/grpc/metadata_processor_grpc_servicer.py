"""GRPC Servicer for Metadata Processor"""

import logging

import grpc
from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import ( # noqa: E402, E501
    PostTrainingMetadataRequest,
    PostTrainingMetadataResponse,
)
from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2_grpc import ( # noqa: E402, E501
    MetadataProcessorServicer,
)
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy

logger = logging.getLogger(__name__)


class MetadataProcessorGRPCServicer(MetadataProcessorServicer):
    """GRPC Servicer for the MetadataProcessor module."""

    def __init__(self, strategy: AbstractProcessorStrategy) -> None:
        super().__init__()
        self.__config = config
        self.processor_strategy = strategy

    def ProcessPostTrainingMetadata(
        self, request: PostTrainingMetadataRequest, context: grpc.ServicerContext
    ) -> PostTrainingMetadataResponse:
        """Process post training metadata.

        Args:
            request (PostTrainingMetadataRequest): Request containing the training ID
                and the metadata
            context (grpc.ServicerContext): Context of the request.

        Returns:
            response (PostTrainingMetadataResponse): Empty response, to confirm.
        """
        logger.info(f"Processing post-training metadata for training ID {request.training_id}")
        self.processor_strategy.process_post_training_metadata(request.training_id, request.data)
        return PostTrainingMetadataResponse()
