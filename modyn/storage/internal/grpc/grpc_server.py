"""GRPC server context manager."""

import logging
from concurrent import futures

import grpc
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import add_StorageServicer_to_server
from modyn.storage.internal.grpc.storage_grpc_servicer import StorageGRPCServicer
from modyn.utils import MAX_MESSAGE_SIZE

logger = logging.getLogger(__name__)


class GRPCServer:
    """GRPC server context manager."""

    def __init__(self, modyn_config: dict) -> None:
        """Initialize the GRPC server.

        Args:
            modyn_config (dict): Configuration of the storage module.
        """
        self.modyn_config = modyn_config
        self.server = grpc.server(
            futures.ThreadPoolExecutor(
                max_workers=64,
            ),
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ],
        )

    def __enter__(self) -> grpc.Server:
        """Enter the context manager.

        Returns:
            grpc.Server: GRPC server
        """
        add_StorageServicer_to_server(StorageGRPCServicer(self.modyn_config), self.server)
        port = self.modyn_config["storage"]["port"]
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
