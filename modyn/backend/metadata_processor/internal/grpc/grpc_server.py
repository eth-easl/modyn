"""GRPC Server Context Manager"""

import logging
from concurrent import futures

import grpc

from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2_grpc import (
    add_MetadataProcessorServicer_to_server,
)
from modyn.backend.metadata_processor.internal.grpc.metadata_processor_grpc_servicer import (
    MetadataProcessorGRPCServicer,
)

logger = logging.getLogger(__name__)


class GRPCServer:
    """GRPC Server Context Manager"""

    def __init__(self, config: dict) -> None:
        """Initialize the GRPC server.

        Args:
            config (dict): configuration of the metadata processor module
        """
        self.config = config
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    def __enter__(self) -> grpc.Server:
        """Enter the context manager.

        Returns:
            grpc.Server: GRPC server
        """
        strategy = None  # TODO: get custom strategy
        add_MetadataProcessorServicer_to_server(
            MetadataProcessorGRPCServicer(self.config, strategy), self.server
        )

        port = self.config["metadata_processor"]["port"]
        logger.info(f"Starting server. Listening on port {port}")
        self.server.add_insecure_port("[::]:" + port)
        self.server.start()

        return self.server

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        """Exit the context manager.

        Args:
            exc_type (type): exception type
            exc_val (Exception): exception value
            exc_tb (Exception): exception traceback
        """
        self.server.stop(0)
