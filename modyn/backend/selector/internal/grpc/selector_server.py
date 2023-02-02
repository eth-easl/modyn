import logging
from concurrent import futures

import grpc
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import (  # noqa: E402, E501
    add_SelectorServicer_to_server,
)
from modyn.backend.selector.internal.grpc.selector_grpc_servicer import SelectorGRPCServicer
from modyn.backend.selector.internal.selector_manager import SelectorManager

logger = logging.getLogger(__name__)


class SelectorServer:
    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config
        self.selector_manager = SelectorManager(modyn_config)
        self.grpc_servicer = SelectorGRPCServicer(self.selector_manager)
        self._add_servicer_to_server_func = add_SelectorServicer_to_server

    def prepare_server(self) -> grpc.server:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self._add_servicer_to_server_func(self.grpc_servicer, server)
        return server

    def run(self) -> None:
        server = self.prepare_server()
        logger.info(f"Starting server. Listening on port {self.modyn_config['selector']['port']}.")
        server.add_insecure_port("[::]:" + self.modyn_config["selector"]["port"])
        server.start()
        server.wait_for_termination()
