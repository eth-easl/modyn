import json
from typing import Tuple

import grpc

# pylint: disable-next=no-name-in-module
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import (
    GetSelectionStrategyRequest,
    SelectionStrategyResponse,
)
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.utils import grpc_connection_established


def get_selection_strategy(selector_address: str, pipeline_id: int) -> Tuple[bool, str, dict]:
    assert selector_address is not None
    selector_stub = _init_grpc(selector_address)

    req = GetSelectionStrategyRequest(pipeline_id=pipeline_id)

    response: SelectionStrategyResponse = selector_stub.get_selection_strategy(req)

    params = json.loads(response.params.value)

    return response.downsampling_enabled, response.strategy_name, params


def _init_grpc(selector_address: str) -> SelectorStub:
    selector_channel = grpc.insecure_channel(selector_address)
    assert selector_channel is not None
    if not grpc_connection_established(selector_channel):
        raise ConnectionError(f"Could not establish gRPC connection to selector at address {selector_address}.")
    return SelectorStub(selector_channel)
