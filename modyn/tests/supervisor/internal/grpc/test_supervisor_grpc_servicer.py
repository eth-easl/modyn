# pylint: disable=unused-argument, no-name-in-module, redefined-outer-name
import json
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from modyn.config.schema.system.config import ModynConfig, SupervisorConfig
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import (  # noqa: E402, E501, E611;
    GetPipelineStatusRequest,
    GetPipelineStatusResponse,
    JsonString,
    PipelineResponse,
    StartPipelineRequest,
)
from modyn.supervisor.internal.grpc.supervisor_grpc_servicer import SupervisorGRPCServicer
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.supervisor import Supervisor

EVALUATION_DIRECTORY: pathlib.Path = pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir"


@pytest.fixture
def minimal_system_config(dummy_system_config: ModynConfig) -> ModynConfig:
    config = dummy_system_config.model_copy()
    config.supervisor = SupervisorConfig(hostname="localhost", port=50051, eval_directory=EVALUATION_DIRECTORY)
    return config


def noop_constructor_mock(self, modyn_config: dict) -> None:
    pass


def noop(self) -> None:
    pass


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


def noop_init_metadata_db(self) -> None:
    pass


@patch.object(GRPCHandler, "__init__", noop_constructor_mock)
@patch.object(GRPCHandler, "init_cluster_connection", noop)
@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
def test_init(minimal_system_config: ModynConfig):
    sup = Supervisor(minimal_system_config)
    servicer = SupervisorGRPCServicer(sup, minimal_system_config.model_dump(by_alias=True))
    assert servicer._supervisor == sup


@patch.object(GRPCHandler, "__init__", noop_constructor_mock)
@patch.object(GRPCHandler, "init_cluster_connection", noop)
@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
@patch.object(Supervisor, "start_pipeline", return_value={"pipeline_id": 1})
def test_start_pipeline(test_start_pipeline: MagicMock, minimal_system_config: ModynConfig):
    sup = Supervisor(minimal_system_config)
    servicer = SupervisorGRPCServicer(sup, minimal_system_config.model_dump(by_alias=True))

    pipeline_config = get_minimal_pipeline_config()
    request = StartPipelineRequest(
        pipeline_config=JsonString(value=json.dumps(pipeline_config)),
        start_replay_at=0,
        stop_replay_at=1,
        maximum_triggers=2,
    )

    response: PipelineResponse = servicer.start_pipeline(request, None)
    assert response.pipeline_id == 1

    test_start_pipeline.assert_called_once_with(
        pipeline_config, minimal_system_config.supervisor.eval_directory, 0, 1, 2
    )


@patch.object(GRPCHandler, "__init__", noop_constructor_mock)
@patch.object(GRPCHandler, "init_cluster_connection", noop)
@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
@patch.object(
    Supervisor,
    "get_pipeline_status",
    return_value={
        "status": "running",
        "pipeline_stage": [{"stage": "test", "msg_type": "test", "log": False}],
    },
)
def test_get_pipeline_status(test_get_pipeline_status: MagicMock, minimal_system_config: ModynConfig):
    sup = Supervisor(minimal_system_config)
    servicer = SupervisorGRPCServicer(sup, minimal_system_config.model_dump(by_alias=True))

    request = GetPipelineStatusRequest(pipeline_id=42)

    response: GetPipelineStatusResponse = servicer.get_pipeline_status(request, None)
    assert response.status == "running"

    test_get_pipeline_status.assert_called_once_with(42)
