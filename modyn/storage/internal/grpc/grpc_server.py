import logging
from concurrent import futures

import grpc

from modyn.storage.internal.grpc.generated.storage_pb2_grpc import add_StorageServicer_to_server
from modyn.storage.internal.grpc.storage_grpc_servicer import StorageGRPCServicer

logger = logging.getLogger(__name__)


class GRPCServer():

    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    def __enter__(self) -> grpc.Server:
        add_StorageServicer_to_server(
            StorageGRPCServicer(self.modyn_config), self.server)
        port = self.modyn_config['storage']['port']
        logger.info(f'Starting server. Listening on port {port}')
        self.server.add_insecure_port('[::]:' + port)
        self.server.start()
        return self.server

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        self.server.stop(0)
