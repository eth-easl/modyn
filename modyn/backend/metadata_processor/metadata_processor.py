"""Metadata Processor module.

The Metadata Processor module contains all classes and functions related to the
post-training processing of metadata collected by the GPU Node.
"""

import logging

from modyn.backend.metadata_processor.internal.grpc.grpc_server import GRPCServer
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy
from modyn.backend.metadata_processor.processor_strategies.basic_processor_strategy import BasicProcessorStrategy

logger = logging.getLogger(__name__)


class MetadataProcessor:
    def __init__(self, modyn_config: dict, pipeline_config: dict) -> None:
        self.config = modyn_config
        self.strategy = self._get_strategy(pipeline_config)

    def run(self) -> None:
        with GRPCServer(self.config, self.strategy) as server:
            server.wait_for_termination()

    def _get_strategy(self, pipeline_config: dict) -> MetadataProcessorStrategy:
        # TODO: improve this
        strategy_name = pipeline_config["training"]["strategy"]
        if strategy_name == "finetune":
            return BasicMetadataProcessor(self.config)
        raise NotImplementedError(f"{strategy_name} is not implemented")
