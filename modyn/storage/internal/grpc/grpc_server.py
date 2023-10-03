"""GRPC server context manager."""


import logging
from typing import Any

from modyn.common.grpc import GenericGRPCServer
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import add_StorageServicer_to_server
from modyn.storage.internal.grpc.storage_grpc_servicer import StorageGRPCServicer

logger = logging.getLogger(__name__)


class StorageGRPCServer(GenericGRPCServer):
    """GRPC server context manager."""

    @staticmethod
    def callback(modyn_config: dict, server: Any) -> None:
        add_StorageServicer_to_server(StorageGRPCServicer(modyn_config), server)

    def __init__(self, modyn_config: dict) -> None:
        """Initialize the GRPC server.

        Args:
            modyn_config (dict): Configuration of the storage module.
        """
        super().__init__(modyn_config, modyn_config["storage"]["port"], StorageGRPCServer.callback)
