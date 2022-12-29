import logging
from concurrent import futures

import grpc

from modyn.storage.internal.grpc.generated.storage_pb2_grpc import add_StorageServicer_to_server
from modyn.storage.internal.grpc.storage_grpc_server import StorageGRPCServer

logger = logging.getLogger(__name__)


class GRPCServer():
    server: grpc.Server = None

    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config

    def __enter__(self) -> grpc.Server:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_StorageServicer_to_server(
            StorageGRPCServer(self.modyn_config), server)
        port = self.modyn_config['storage']['port']
        logger.info(f'Starting server. Listening on port {port}')
        server.add_insecure_port('[::]:' + self.modyn_config['storage']['port'])
        server.start()
        self.server = server
        return server

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        if self.server is not None:
            for thread in self.server._state._threads:
                thread.join()
            self.server.stop(0)
