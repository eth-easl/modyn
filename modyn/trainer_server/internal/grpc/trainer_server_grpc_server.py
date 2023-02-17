"""GRPC server context manager."""

import logging
import pathlib
import shutil
import tempfile
from concurrent import futures

import grpc
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2_grpc import add_TrainerServerServicer_to_server
from modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer import TrainerServerGRPCServicer

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024


class GRPCServer:
    """GRPC server context manager."""

    def __init__(self, config: dict) -> None:
        """Initialize the GRPC server.

        Args:
            config (dict): Modyn configuration.
        """
        self.config = config
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
                ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ],
        )

    def __enter__(self) -> grpc.Server:
        """Enter the context manager.

        Returns:
            grpc.Server: GRPC server
        """

        add_TrainerServerServicer_to_server(TrainerServerGRPCServicer(self.config), self.server)
        logger.info(f"Starting trainer server. Listening on port {self.config['trainer_server']['port']}")
        self.server.add_insecure_port("[::]:" + self.config["trainer_server"]["port"])
        logger.info("start serving!")
        self.server.start()
        return self.server

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        """Exit the context manager.

        Args:
            exc_type (type): exception type
            exc_val (Exception): exception value
            exc_tb (Exception): exception traceback
        """

        shutil.rmtree(pathlib.Path(tempfile.gettempdir()) / "modyn")
        self.server.stop(0)
