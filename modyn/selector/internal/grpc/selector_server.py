import contextlib
import datetime
import logging
import multiprocessing as mp
import os
import socket
import time
from concurrent import futures

import grpc
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import add_SelectorServicer_to_server  # noqa: E402, E501
from modyn.selector.internal.grpc.selector_grpc_servicer import SelectorGRPCServicer
from modyn.selector.internal.selector_manager import SelectorManager
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


def _run_server(bind_address, selector_manager, sample_batch_size):
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
    add_SelectorServicer_to_server(SelectorGRPCServicer(selector_manager, sample_batch_size), server)
    server.add_insecure_port(bind_address)
    server.start()
    _wait_forever(server)


class SelectorServer:
    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config
        self.selector_manager = SelectorManager(modyn_config)
        self.sample_batch_size = self.modyn_config["selector"]["sample_batch_size"]
        self.workers = []

    def run(self) -> None:
        port = self.modyn_config["selector"]["port"]
        logger.info(f"Starting server. Listening on port {port}")
        with _reserve_port(port) as port:
            bind_address = "[::]:" + port
            for _ in range(64):
                worker = mp.Process(
                    target=_run_server,
                    args=(bind_address, self.selector_manager, self.sample_batch_size),
                )
                worker.start()
                self.workers.append(worker)

            for worker in self.workers:
                worker.join()

        if (
            "cleanup_trigger_samples_after_shutdown" in self.modyn_config["selector"]
            and self.modyn_config["selector"]["cleanup_trigger_samples_after_shutdown"]
        ):
            self.selector_manager.cleanup_trigger_samples()
