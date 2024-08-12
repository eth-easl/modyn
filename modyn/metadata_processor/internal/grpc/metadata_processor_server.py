"""GRPC Server Context Manager."""

import logging
from concurrent import futures

import grpc

from modyn.metadata_processor.internal.grpc.generated.metadata_processor_pb2_grpc import (
    add_MetadataProcessorServicer_to_server,
)
from modyn.metadata_processor.internal.grpc.metadata_processor_grpc_servicer import MetadataProcessorGRPCServicer
from modyn.metadata_processor.internal.metadata_processor_manager import MetadataProcessorManager

logger = logging.getLogger(__name__)


class MetadataProcessorServer:
    """GRPC Server Context Manager."""

    def __init__(self, modyn_config: dict) -> None:
        self.config = modyn_config
        self.processor_manager = MetadataProcessorManager(modyn_config)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self._add_servicer_to_server_func = add_MetadataProcessorServicer_to_server

    def prepare_server(self) -> grpc.server:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self._add_servicer_to_server_func(MetadataProcessorGRPCServicer(self.processor_manager), server)
        return server

    def run(self) -> None:
        server = self.prepare_server()

        port = self.config["metadata_processor"]["port"]
        logger.info(f"Starting server. Listening on port {port}")

        server.add_insecure_port("[::]:" + port)
        server.start()
        server.wait_for_termination()
