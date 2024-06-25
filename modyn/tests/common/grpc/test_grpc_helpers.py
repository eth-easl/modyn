import json
import multiprocessing as mp
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from modyn.common.grpc import GenericGRPCServer
from modyn.common.grpc.grpc_helpers import TrainerServerGRPCHandlerMixin
from modyn.config import (
    CheckpointingConfig,
    DataConfig,
    LrSchedulerConfig,
    ModynConfig,
    ModynPipelineConfig,
    TrainingConfig,
)
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


@pytest.fixture()
def pipeline_data_config():
    return DataConfig(
        dataset_id="test",
        bytes_parser_function="def bytes_parser_function(x):\n\treturn x",
        label_transformer_function="def label_transformer_function(x):\n\treturn x",
        transformations=["transformation1", "transformation2"],
    )


@pytest.fixture()
def lr_scheduler_config():
    return LrSchedulerConfig(
        name="CosineAnnealingLR",
        source="PyTorch",
        step_every="batch",
        optimizers=["default"],
        config={"T_max": "MODYN_NUM_BATCHES", "eta_min": 0.001},
    )


@pytest.mark.parametrize("previous_model_id", [1, None])
@pytest.mark.parametrize("num_samples_to_pass", [5, None])
@pytest.mark.parametrize("set_lr_scheduler_to_none", [True, False])
@pytest.mark.parametrize("disable_checkpointing", [True, False])
def test_prepare_start_training_request(
    disable_checkpointing: bool,
    set_lr_scheduler_to_none: bool,
    num_samples_to_pass: Optional[int],
    previous_model_id: Optional[int],
    pipeline_training_config: TrainingConfig,
    pipeline_data_config: DataConfig,
    lr_scheduler_config: LrSchedulerConfig,
):
    # for bool value False is the default value so we don't need to test it
    pipeline_training_config.shuffle = True
    pipeline_training_config.enable_accurate_gpu_measurements = True
    # for int value 0 is the default value so we don't need to test it
    pipeline_training_config.record_loss_every = 10
    pipeline_training_config.optimization_criterion.config = {"key": "value"}
    pipeline_training_config.use_previous_model = previous_model_id is not None

    pipeline_training_config.lr_scheduler = None if set_lr_scheduler_to_none else lr_scheduler_config
    if set_lr_scheduler_to_none:
        expected_lr_scheduler_config = {}
    else:
        expected_lr_scheduler_config = lr_scheduler_config.model_dump(by_alias=True)
    if disable_checkpointing:
        pipeline_training_config.checkpointing = CheckpointingConfig(activated=False)
    else:
        pipeline_training_config.checkpointing = CheckpointingConfig(activated=True, interval=1, path=Path("test"))

    pipeline_id = 42
    trigger_id = 21

    req = TrainerServerGRPCHandlerMixin.prepare_start_training_request(
        pipeline_id, trigger_id, pipeline_training_config, pipeline_data_config, previous_model_id, num_samples_to_pass
    )

    assert req.pipeline_id == pipeline_id
    assert req.trigger_id == trigger_id
    assert req.device == pipeline_training_config.device
    assert req.use_pretrained_model == pipeline_training_config.use_previous_model
    assert req.pretrained_model_id == previous_model_id if previous_model_id is not None else -1
    assert req.batch_size == pipeline_training_config.batch_size
    assert req.torch_criterion == pipeline_training_config.optimization_criterion.name
    assert json.loads(req.criterion_parameters.value) == pipeline_training_config.optimization_criterion.config
    assert req.data_info.dataset_id == pipeline_data_config.dataset_id
    assert req.data_info.num_dataloaders == pipeline_training_config.dataloader_workers
    if disable_checkpointing:
        assert req.checkpoint_info.checkpoint_path == ""
        assert req.checkpoint_info.checkpoint_interval == 0
    else:
        assert req.checkpoint_info.checkpoint_path == str(pipeline_training_config.checkpointing.path)
        assert req.checkpoint_info.checkpoint_interval == pipeline_training_config.checkpointing.interval
    assert req.bytes_parser.value == pipeline_data_config.bytes_parser_function
    assert req.transform_list == pipeline_data_config.transformations
    assert req.label_transformer.value == pipeline_data_config.label_transformer_function
    assert json.loads(req.lr_scheduler.value) == expected_lr_scheduler_config
    assert req.epochs_per_trigger == pipeline_training_config.epochs_per_trigger
    assert req.num_prefetched_partitions == pipeline_training_config.num_prefetched_partitions
    assert req.parallel_prefetch_requests == pipeline_training_config.parallel_prefetch_requests
    assert req.seed == 0
    assert req.num_samples_to_pass == (num_samples_to_pass if num_samples_to_pass is not None else 0)
    assert req.shuffle
    assert req.enable_accurate_gpu_measurements
    assert req.record_loss_every == 10


@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
@patch.object(
    TrainerServerGRPCHandlerMixin,
    "prepare_start_training_request",
    wraps=TrainerServerGRPCHandlerMixin.prepare_start_training_request,
)
def test_start_training(
    test_prepare_start_training_request: MagicMock,
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
        test_prepare_start_training_request.assert_called_once_with(
            pipeline_id, trigger_id, dummy_pipeline_config.training, dummy_pipeline_config.data, None, None
        )


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
