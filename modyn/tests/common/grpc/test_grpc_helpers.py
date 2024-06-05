import multiprocessing as mp
from unittest.mock import MagicMock, patch

from modyn.common.grpc import GenericGRPCServer
from modyn.common.grpc.grpc_helpers import TrainerServerGRPCHandlerMixin
from modyn.config import ModynConfig, ModynPipelineConfig
from modyn.supervisor.internal.utils import TrainingStatusReporter
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import JsonString as TrainerJsonString
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    StartTrainingResponse,
    StoreFinalModelRequest,
    StoreFinalModelResponse,
    TrainerAvailableResponse,
    TrainingStatusResponse,
)

# TODO(310): add more meaningful tests


def test_init():
    GenericGRPCServer({}, "1234", lambda x: None)


@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_init_and_trainer_server_available(
    test_grpc_connection_established: MagicMock,
    dummy_system_config: ModynConfig,
):
    handler = TrainerServerGRPCHandlerMixin(dummy_system_config.model_dump(by_alias=True))
    assert handler.trainer_server is None
    assert not handler.connected_to_trainer_server

    handler.init_trainer_server()
    assert handler.trainer_server is not None
    assert handler.connected_to_trainer_server

    with patch.object(
        handler.trainer_server, "trainer_available", return_value=TrainerAvailableResponse(available=True)
    ) as avail_method:
        assert handler.trainer_server_available()
        avail_method.assert_called_once()


@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_start_training(
    test_common_connection_established: MagicMock,
    dummy_pipeline_config: ModynPipelineConfig,
    dummy_system_config: ModynConfig,
):
    handler = TrainerServerGRPCHandlerMixin(dummy_system_config.model_dump(by_alias=True))
    handler.init_trainer_server()

    pipeline_id = 42
    trigger_id = 21

    with patch.object(
        handler.trainer_server,
        "start_training",
        return_value=StartTrainingResponse(training_started=True, training_id=42),
    ) as avail_method:
        assert (
            handler.start_training(
                pipeline_id, trigger_id, dummy_pipeline_config.training, dummy_pipeline_config.data, None
            )
            == 42
        )
        avail_method.assert_called_once()


@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_wait_for_training_completion(
    test_common_connection_established: MagicMock,
    dummy_system_config: ModynConfig,
):
    # This test primarily checks whether we terminate.
    handler = TrainerServerGRPCHandlerMixin(dummy_system_config.model_dump(by_alias=True))
    handler.init_trainer_server()

    with patch.object(
        handler.trainer_server,
        "get_training_status",
        return_value=TrainingStatusResponse(
            valid=True,
            blocked=False,
            exception=None,
            state_available=False,
            is_running=False,
            log=TrainerJsonString(value='{"a": 1}'),
        ),
    ) as avail_method:
        training_reporter = TrainingStatusReporter(mp.Queue(), 21, 42, 22, 100)
        log = handler.wait_for_training_completion(42, training_reporter)
        avail_method.assert_called_once()
        assert log == {"a": 1}


@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_store_trained_model(
    test_common_connection_established: MagicMock,
    dummy_system_config: ModynConfig,
):
    handler = TrainerServerGRPCHandlerMixin(dummy_system_config.model_dump(by_alias=True))
    handler.init_trainer_server()

    res = StoreFinalModelResponse(valid_state=True, model_id=42)

    with patch.object(handler.trainer_server, "store_final_model", return_value=res) as get_method:
        model_id = handler.store_trained_model(21)
        get_method.assert_called_once_with(StoreFinalModelRequest(training_id=21))
        assert model_id == 42
