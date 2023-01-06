import logging
from concurrent import futures

import grpc

from modyn.storage.internal.grpc.generated.storage_pb2_grpc import add_StorageServicer_to_server
from modyn.storage.internal.grpc.storage_grpc_servicer import StorageGRPCServicer

logger = logging.getLogger(__name__)


class GRPCServer():
    server: grpc.Server = None

    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config

    def __enter__(self) -> grpc.Server:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self._add_storage_servicer_to_server(
            StorageGRPCServicer(self.modyn_config), server)
        port = self.modyn_config['storage']['port']
        logger.info(f'Starting server. Listening on port {port}')
        server.add_insecure_port('[::]:' + port)
        server.start()
        self.server = server
        return server

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        if self.server is not None:
            self.server.stop(0)

    def _add_storage_servicer_to_server(self, storage_grpc_server: StorageGRPCServicer, server: grpc.Server) -> None:
        add_StorageServicer_to_server(storage_grpc_server, server)  # pragma: no cover
