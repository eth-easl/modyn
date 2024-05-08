import json
from unittest.mock import patch

import grpc
import pytest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import GetPipelineStatusRequest, GetPipelineStatusResponse
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import JsonString as SupervisorJsonString
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import PipelineResponse, StartPipelineRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorStub
from modynclient.client.internal.grpc_handler import GRPCHandler
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor

PIPELINE_ID = 42
START_PIPELINE_RES = PipelineResponse(pipeline_id=PIPELINE_ID)
PIPELINE_STATUS_RES_RUNNING = GetPipelineStatusResponse(status="running")


def noop_constructor_mock(self, channel: grpc.Channel) -> None:
    pass


def get_simple_config() -> ModynClientConfig:
    return ModynClientConfig(
        supervisor=Supervisor(
            ip="127.0.0.1",
            port=42,
        )
    )


def get_minimal_pipeline_config() -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "training": {
            "gpus": 1,
            "device": "cpu",
            "amp": False,
            "dataloader_workers": 1,
            "initial_model": "random",
            "initial_pass": {"activated": False},
            "batch_size": 42,
            "optimizers": [
                {"name": "default", "algorithm": "SGD", "source": "PyTorch", "param_groups": [{"module": "model"}]},
            ],
            "optimization_criterion": {"name": "CrossEntropyLoss"},
            "checkpointing": {"activated": False},
            "selection_strategy": {"name": "NewDataStrategy"},
        },
        "data": {"dataset_id": "test", "bytes_parser_function": "def bytes_parser_function(x):\n\treturn x"},
        "trigger": {"id": "DataAmountTrigger", "trigger_config": {"data_points_for_trigger": 1}},
        "evaluation": {
            "device": "cpu",
            "datasets": [
                {
                    "dataset_id": "MNIST_eval",
                    "bytes_parser_function": "def bytes_parser_function(data: bytes) -> bytes:\n\treturn data",
                    "dataloader_workers": 2,
                    "batch_size": 64,
                    "metrics": [{"name": "Accuracy"}],
                }
            ],
        },
    }


@patch.object(SupervisorStub, "__init__", noop_constructor_mock)
@patch("modynclient.client.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def get_non_connecting_handler(insecure_channel, init) -> GRPCHandler:
    return GRPCHandler(get_simple_config())


@patch.object(SupervisorStub, "__init__", noop_constructor_mock)
@patch("modynclient.client.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init(test_insecure_channel, test_connection_established):
    handler = GRPCHandler(get_simple_config())
    assert handler.connected_to_supervisor


@patch.object(SupervisorStub, "__init__", noop_constructor_mock)
@patch("modynclient.client.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_supervisor(test_insecure_channel, test_connection_established):
    handler = None
    with patch.object(GRPCHandler, "init_supervisor", return_value=None):
        handler = GRPCHandler(get_simple_config())
    assert handler is not None
    assert not handler.connected_to_supervisor

    handler.init_supervisor()
    assert handler.connected_to_supervisor


@patch.object(SupervisorStub, "__init__", noop_constructor_mock)
@patch("modynclient.client.internal.grpc_handler.grpc_connection_established", return_value=False)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_storage_throws(test_insecure_channel, test_connection_established):
    handler = None

    with patch.object(GRPCHandler, "init_supervisor", return_value=None):
        handler = GRPCHandler(get_simple_config())
    assert handler is not None
    assert not handler.connected_to_supervisor

    with pytest.raises(ConnectionError):
        handler.init_supervisor()


@patch("modynclient.client.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_start_pipeline(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config())
    pipeline_config = get_minimal_pipeline_config()

    req_minimal = StartPipelineRequest(pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)))

    with patch.object(handler.supervisor, "start_pipeline") as mock:
        mock.return_value = START_PIPELINE_RES
        ret_minimal = handler.start_pipeline(pipeline_config)

        assert ret_minimal["pipeline_id"] == 42
        assert "exception" not in ret_minimal
        mock.assert_called_once_with(req_minimal)

    start_replay_at = 0
    stop_replay_at = 1
    maximum_triggers = 10

    req_full = StartPipelineRequest(
        pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
        start_replay_at=start_replay_at,
        stop_replay_at=stop_replay_at,
        maximum_triggers=maximum_triggers,
    )

    with patch.object(handler.supervisor, "start_pipeline") as mock:
        mock.return_value = START_PIPELINE_RES
        ret_full = handler.start_pipeline(
            pipeline_config,
            start_replay_at,
            stop_replay_at,
            maximum_triggers,
        )

        assert ret_full["pipeline_id"] == 42
        assert "exception" not in ret_full
        mock.assert_called_once_with(req_full)


def test_start_pipeline_throws():
    handler = get_non_connecting_handler()
    handler.connected_to_supervisor = False
    with pytest.raises(ConnectionError):
        handler.start_pipeline(get_minimal_pipeline_config())


@patch("modynclient.client.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_get_pipeline_status(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config())

    with patch.object(handler.supervisor, "get_pipeline_status") as mock:
        mock.return_value = PIPELINE_STATUS_RES_RUNNING
        ret = handler.get_pipeline_status(PIPELINE_ID)

        assert ret["status"] == "running"
        mock.assert_called_once_with(GetPipelineStatusRequest(pipeline_id=PIPELINE_ID))


def test_get_pipeline_status_throws():
    handler = get_non_connecting_handler()
    handler.connected_to_supervisor = False
    with pytest.raises(ConnectionError):
        handler.get_pipeline_status(PIPELINE_ID)
