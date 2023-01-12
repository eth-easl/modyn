import logging
import os
import pathlib
from concurrent import futures
from typing import List, Tuple

import grpc
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import (  # noqa: E402, E501
    add_SelectorServicer_to_server,
)
from modyn.backend.selector.internal.grpc.selector_grpc_servicer import (
    SelectorGRPCServicer,
)
from modyn.backend.selector.internal.selector_strategies.data_freshness_strategy import (
    DataFreshnessStrategy,
)
from modyn.backend.selector.selector_strategy import SelectorStrategy
from modyn.utils import validate_yaml

logger = logging.getLogger(__name__)


class Selector:
    def __init__(self, pipeline_config: dict, modyn_config: dict) -> None:
        self.pipeline_config = pipeline_config
        self.modyn_config = modyn_config

        valid, errors = self._validate_pipeline()
        if not valid:
            raise ValueError(f"Invalid configuration: {errors}")

        self.strategy = self._get_strategy()
        self.grpc_server = SelectorGRPCServicer(self.strategy)

    def _validate_pipeline(self) -> Tuple[bool, List[str]]:
        schema_path = (
            pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "schema" / "pipeline-schema.yaml"
        )
        return validate_yaml(self.pipeline_config, schema_path)

    def run(self) -> None:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        add_SelectorServicer_to_server(self.grpc_server, server)
        logging.info(f"Starting server. Listening on port {self.modyn_config['selector']['port']}.")
        server.add_insecure_port("[::]:" + self.modyn_config["selector"]["port"])
        server.start()
        server.wait_for_termination()

    def _get_strategy(self) -> SelectorStrategy:
        strategy_name = self.pipeline_config["training"]["strategy"]
        if strategy_name == "finetune":
            config = {"selector": {"unseen_data_ratio": 1.0, "is_adaptive_ratio": False}}
            return DataFreshnessStrategy(config)
        raise NotImplementedError(f"{strategy_name} is not implemented")
