"""GRPC server context manager."""

import contextlib
import datetime
import logging
import multiprocessing as mp
import os
import socket
import time
from concurrent import futures
from typing import Any

import grpc
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import add_StorageServicer_to_server
from modyn.storage.internal.grpc.storage_grpc_servicer import StorageGRPCServicer
from modyn.utils import MAX_MESSAGE_SIZE

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _reserve_port(port: str):
    """Find and reserve a port for all subprocesses to use."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(("", int(port)))
    try:
        assert sock.getsockname()[1] == int(port)
        yield port
    finally:
        sock.close()


def _wait_forever(server):
    try:
        while True:
            time.sleep(datetime.timedelta(days=1).total_seconds())
    except KeyboardInterrupt:
        server.stop(None)


def _run_server(bind_address, modyn_config):
    """Start a server in a subprocess."""
    logging.info(f"[{os.getpid()}] Starting new server.")

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=16,
        ),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.so_reuseport", 1),
        ],
    )
    add_StorageServicer_to_server(StorageGRPCServicer(modyn_config), server)
    server.add_insecure_port(bind_address)
    server.start()
    _wait_forever(server)


class GRPCServer:
    """GRPC server context manager."""

    def __init__(self, modyn_config: dict) -> None:
        """Initialize the GRPC server.

        Args:
            modyn_config (dict): Configuration of the storage module.
        """
        self.modyn_config = modyn_config
        self.workers = []

    def __enter__(self) -> Any:
        """Enter the context manager.

        Returns:
            grpc.Server: GRPC server
        """
        port = self.modyn_config["storage"]["port"]
        logger.info(f"Starting server. Listening on port {port}")
        with _reserve_port(port) as port:
            bind_address = "[::]:" + port
            for _ in range(64):
                worker = mp.Process(
                    target=_run_server,
                    args=(
                        bind_address,
                        self.modyn_config,
                    ),
                )
                worker.start()
                self.workers.append(worker)

        return self

    def wait_for_termination(self) -> None:
        for worker in self.workers:
            worker.join()

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        """Exit the context manager.

        Args:
            exc_type (type): exception type
            exc_val (Exception): exception value
            exc_tb (Exception): exception traceback
        """
        self.wait_for_termination()
        del self.workers
