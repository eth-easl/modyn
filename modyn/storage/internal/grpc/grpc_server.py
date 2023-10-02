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
from modyn.common.grpc import GenericGRPCServer
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import add_StorageServicer_to_server
from modyn.storage.internal.grpc.storage_grpc_servicer import StorageGRPCServicer
from modyn.utils import MAX_MESSAGE_SIZE

logger = logging.getLogger(__name__)


class StorageGRPCServer(GenericGRPCServer):
    """GRPC server context manager."""

    @staticmethod
    def callback(modyn_config, server):
        add_StorageServicer_to_server(StorageGRPCServicer(modyn_config), server)

    def __init__(self, modyn_config: dict) -> None:
        """Initialize the GRPC server.

        Args:
            modyn_config (dict): Configuration of the storage module.
        """
        super().__init__(modyn_config, modyn_config["storage"]["port"], StorageGRPCServer.callback)
