"""GRPC server context manager."""

import logging
import pathlib
from concurrent import futures

import grpc

from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import add_ModelStorageServicer_to_server
from modyn.model_storage.internal.grpc.model_storage_grpc_servicer import ModelStorageGRPCServicer
from modyn.utils import MAX_MESSAGE_SIZE

logger = logging.getLogger(__name__)


class GRPCServer:
    """GRPC server context manager."""

    def __init__(self, modyn_config: dict, storage_dir: pathlib.Path, ftp_directory: pathlib.Path) -> None:
        """Initialize the GRPC server.

        Args:
            modyn_config (dict): Configuration of the storage module.
            storage_dir (path): Path to the model storage directory.
            ftp_directory (path): Path to the ftp directory.
        """
        self.modyn_config = modyn_config
        self.storage_dir = storage_dir
        self.ftp_directory = ftp_directory
        self.server = grpc.server(
            futures.ThreadPoolExecutor(
                max_workers=10,
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
        add_ModelStorageServicer_to_server(
            ModelStorageGRPCServicer(self.modyn_config, self.storage_dir, self.ftp_directory), self.server
        )
        port = self.modyn_config["model_storage"]["port"]
        logger.info(f"Starting GRPC server. Listening on port {port}")
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
