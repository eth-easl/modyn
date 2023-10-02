import contextlib
import datetime
import logging
import multiprocessing as mp
import os
import pickle
import socket
import time
from concurrent import futures
from typing import Any, Callable

import grpc
from modyn.utils import MAX_MESSAGE_SIZE

logger = logging.getLogger(__name__)

PROCESS_THREAD_WORKERS = 16
NUM_GPRC_PROCESSES = 64


@contextlib.contextmanager
def reserve_port(port: str):
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


def _run_server_worker(bind_address: str, add_servicer_callback: Callable, modyn_config: dict, callback_kwargs: dict):
    """Start a server in a subprocess."""
    logging.info(f"[{os.getpid()}] Starting new gRPC server process.")

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=PROCESS_THREAD_WORKERS,
        ),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.so_reuseport", 1),
        ],
    )

    add_servicer_callback(modyn_config, server, **callback_kwargs)
    server.add_insecure_port(bind_address)
    server.start()
    _wait_forever(server)


class GenericGRPCServer:
    def __init__(
        self, modyn_config: dict, port: str, add_servicer_callback: Callable, callback_kwargs: dict = {}
    ) -> None:
        """Initialize the GRPC server.

        Args:
            TODO
        """
        self.port = port
        self.modyn_config = modyn_config
        self.add_servicer_callback = add_servicer_callback
        self.callback_kwargs = callback_kwargs
        self.workers = []

    def __enter__(self) -> Any:
        """Enter the context manager.

        Returns:
            grpc.Server: GRPC server
        """
        logger.info(f"[{os.getpid()}] Starting server. Listening on port {self.port}")
        with reserve_port(self.port) as port:
            bind_address = "[::]:" + port
            for _ in range(NUM_GPRC_PROCESSES):
                worker = mp.Process(
                    target=_run_server_worker,
                    args=(bind_address, self.add_servicer_callback, self.modyn_config, self.callback_kwargs),
                )
                worker.start()
                self.workers.append(worker)

        return self

    def __getstate__(self):
        for variable_name, value in vars(self).items():
            try:
                pickle.dumps(value)
            except:
                print(f"{variable_name} with value {value} is not pickable")

        state = self.__dict__.copy()
        del state["add_servicer_callback"]
        return state

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
