"""Metadata Processor module.

The Metadata Processor module contains all classes and functions related to the
post-training processing of metadata collected by the GPU Node.
"""

import logging

from modyn.backend.metadata_processor.internal.grpc.grpc_server import GRPCServer
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy
from modyn.backend.metadata_processor.processor_strategies.processor_strategy_type import ProcessorStrategyType
from modyn.utils import dynamic_module_import

logger = logging.getLogger(__name__)


class MetadataProcessor:
    def __init__(self, modyn_config: dict, pipeline_config: dict) -> None:
        self.config = modyn_config
        self.strategy = self._get_strategy(pipeline_config)

    def run(self) -> None:
        with GRPCServer(self.config, self.strategy) as server:
            server.wait_for_termination()

    def _get_strategy(self, pipeline_config: dict) -> AbstractProcessorStrategy:
        strategy = ProcessorStrategyType(pipeline_config["training"]["strategy_config"]["processor_type"])
        processor_strategy_module = dynamic_module_import(
            f"modyn.backend.metadata_processor.processor_strategies.{strategy.value}"
        )
        processor_strategy = getattr(processor_strategy_module, f"{strategy.name}")
        return processor_strategy(self.config)
