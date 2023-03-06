"""GRPC server context manager."""

import logging
import pathlib
from concurrent import futures

import grpc
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2_grpc import add_TrainerServerServicer_to_server
from modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer import TrainerServerGRPCServicer

logger = logging.getLogger(__name__)


class GRPCServer:
    """GRPC server context manager."""

    def __init__(self, config: dict, tempdir: pathlib.Path) -> None:
        """Initialize the GRPC server.

        Args:
            config (dict): Modyn configuration.
        """
        self.config = config
        self.tempdir = tempdir
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    def __enter__(self) -> grpc.Server:
        """Enter the context manager.

        Returns:
            grpc.Server: GRPC server
        """

        add_TrainerServerServicer_to_server(TrainerServerGRPCServicer(self.config, self.tempdir), self.server)
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
        self.server.stop(0)
