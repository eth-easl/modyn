# pylint: disable=unused-argument, no-name-in-module, redefined-outer-name
import json
import os
import pathlib
from unittest.mock import MagicMock, patch

from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import (  # noqa: E402, E501, E611;
    JsonString,
    PipelineResponse,
    StartPipelineRequest,
)
from modyn.supervisor.internal.grpc.supervisor_grpc_servicer import SupervisorGRPCServicer
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.supervisor import Supervisor

EVALUATION_DIRECTORY: str = str(pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir")


def get_minimal_modyn_config():
    return {}


def noop_constructor_mock(self, modyn_config: dict) -> None:
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


def noop_init_metadata_db(self) -> None:
    pass


@patch.object(GRPCHandler, "__init__", noop_constructor_mock)
@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
def test_init():
    modyn_config = get_minimal_modyn_config()
    sup = Supervisor(modyn_config)
    servicer = SupervisorGRPCServicer(sup, modyn_config)
    assert servicer._supervisor == sup


@patch.object(GRPCHandler, "__init__", noop_constructor_mock)
@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
@patch.object(Supervisor, "start_pipeline")
def test_start_pipeline(test_start_pipeline: MagicMock):
    modyn_config = get_minimal_modyn_config()
    sup = Supervisor(modyn_config)
    servicer = SupervisorGRPCServicer(sup, modyn_config)

    pipeline_config = get_minimal_pipeline_config()
    request = StartPipelineRequest(
        pipeline_config=JsonString(value=json.dumps(pipeline_config)),
        eval_directory=EVALUATION_DIRECTORY,
        start_replay_at=0,
        stop_replay_at=1,
        maximum_triggers=2,
    )
    test_start_pipeline.return_value = 1

    response: PipelineResponse = servicer.start_pipeline(request, None)
    assert response.pipeline_id == 1

    test_start_pipeline.assert_called_once_with(pipeline_config, EVALUATION_DIRECTORY, 0, 1, 2)
