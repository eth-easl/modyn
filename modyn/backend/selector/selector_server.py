import logging
import os
import pathlib
from concurrent import futures
from typing import List, Tuple

import grpc
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import (  # noqa: E402, E501
    add_SelectorServicer_to_server,
)
from modyn.backend.selector.internal.grpc.selector_grpc_servicer import SelectorGRPCServicer
from modyn.backend.selector.internal.selector_manager import SelectorManager
from modyn.utils import validate_yaml

logger = logging.getLogger(__name__)


class SelectorServer:
    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config
        self.selector_manager = SelectorManager(modyn_config)
        self.grpc_server = SelectorGRPCServicer(self.selector_manager)

    def run(self) -> None:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        add_SelectorServicer_to_server(self.grpc_server, server)
        logger.info(f"Starting server. Listening on port {self.modyn_config['selector']['port']}.")
        server.add_insecure_port("[::]:" + self.modyn_config["selector"]["port"])
        server.start()
        server.wait_for_termination()
