import os
import pathlib
from unittest.mock import patch

import grpc
import pytest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorStub
from modynclient.client.internal.grpc_handler import GRPCHandler

EVALUATION_DIRECTORY: pathlib.Path = pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir"
PIPELINE_ID = 42


def noop_constructor_mock(self, channel: grpc.Channel) -> None:
    pass


def get_simple_config() -> dict:
    return {
        "supervisor": {"ip": "127.0.0.1", "port": 42},
    }


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


def test_start_pipeline():
    pass


def test_start_pipeline_throws():
    handler = get_non_connecting_handler()
    handler.connected_to_supervisor = False
    with pytest.raises(ConnectionError):
        handler.start_pipeline(get_minimal_pipeline_config(), EVALUATION_DIRECTORY)


def test_get_pipeline_status():
    pass


def test_get_pipeline_status_throws():
    handler = get_non_connecting_handler()
    handler.connected_to_supervisor = False
    with pytest.raises(ConnectionError):
        handler.get_pipeline_status(PIPELINE_ID)
