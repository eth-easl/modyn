"""Metadata Processor module.

The Metadata Processor module contains all classes and functions related to the
post-training processing of metadata collected by the GPU Node.
"""

import logging

from modyn.backend.metadata_processor.internal.grpc.grpc_server import GRPCServer

logger = logging.getLogger(__name__)


class MetadataProcessor:
    def __init__(self, config: dict) -> None:
        self.config = config

        # TODO: validate config?

    def run(self) -> None:
        with GRPCServer(self.config) as server:
            server.wait_for_termination()
