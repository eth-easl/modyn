# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
# ruff: noqa: N802  # grpc functions are not snake case

import json
import multiprocessing as mp
import os
import pathlib
import platform
import tempfile
from io import BytesIO
from time import sleep
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import (
    FetchModelRequest,
    FetchModelResponse,
    RegisterModelRequest,
    RegisterModelResponse,
)
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    CheckpointInfo,
    Data,
    GetLatestModelRequest,
    JsonString,
    PythonString,
    StartTrainingRequest,
    StoreFinalModelRequest,
    StoreFinalModelResponse,
    TrainerAvailableRequest,
    TrainingStatusRequest,
)
from modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer import TrainerServerGRPCServicer
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo
from modyn.trainer_server.internal.utils.training_process_info import TrainingProcessInfo
from modyn.utils import calculate_checksum

DATABASE = pathlib.Path(os.path.abspath(__file__)).parent / "test_trainer_server.database"

trainer_available_request = TrainerAvailableRequest()
get_status_request = TrainingStatusRequest(training_id=1)
store_final_model_request = StoreFinalModelRequest(training_id=1)
get_latest_model_request = GetLatestModelRequest(training_id=1)

modyn_config = {
    "trainer_server": {
        "hostname": "trainer_server",
        "port": "5001",
        "ftp_port": "3001",
        "offline_dataset_directory": "/tmp/offline_dataset",
    },
    "metadata_database": {
        "drivername": "sqlite",
        "username": "",
        "password": "",
        "hostname": "",
        "port": 0,
        "database": f"{DATABASE}",
    },
    "storage": {"hostname": "storage", "port": "5002"},
    "selector": {"hostname": "selector", "port": "5003"},
    "model_storage": {"hostname": "model_storage", "port": "5004"},
}


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    DATABASE.unlink(True)

    with MetadataDatabaseConnection(modyn_config) as database:
        database.create_tables()

        database.register_pipeline(
            1,
            "model",
            json.dumps({}),
            True,
            "{}",
            "{}",
            ModelStorageStrategyConfig(name="PyTorchFullModel"),
            incremental_model_strategy=None,
            full_model_interval=None,
        )
    yield

    DATABASE.unlink()


class DummyModelStorageStub:
    # pylint: disable-next=invalid-name
    def FetchModel(self, request: FetchModelRequest) -> FetchModelResponse:
        if request.model_id <= 10:
            return FetchModelResponse(success=True, model_path="testpath.modyn")
        return FetchModelResponse(success=False)

    # pylint: disable-next=invalid-name
    def RegisterModel(self, request: RegisterModelRequest) -> RegisterModelResponse:
        return RegisterModelResponse(success=True, model_id=1)


class DummyModelWrapper:
    def __init__(self, model_configuration=None) -> None:
        self.model = None


def noop():
    return


def get_training_process_info():
    status_query_queue_training = mp.Queue()
    status_response_queue_training = mp.Queue()
    status_query_queue_downsampling = mp.Queue()
    status_response_queue_downsampling = mp.Queue()
    exception_queue = mp.Queue()

    training_process_info = TrainingProcessInfo(
        mp.Process(),
        exception_queue,
        status_query_queue_training,
        status_response_queue_training,
        status_query_queue_downsampling,
        status_response_queue_downsampling,
    )
    return training_process_info


def get_start_training_request(checkpoint_path=""):
    return StartTrainingRequest(
        pipeline_id=1,
        trigger_id=1,
        device="cpu",
        batch_size=32,
        torch_optimizers_configuration=JsonString(
            value=json.dumps(
                {
                    "default": {
                        "algorithm": "SGD",
                        "param_groups": [{"module": "model", "config": {"lr": 0.1}}],
                    }
                }
            )
        ),
        torch_criterion="CrossEntropyLoss",
        criterion_parameters=JsonString(value=json.dumps({})),
        data_info=Data(dataset_id="Dataset", num_dataloaders=1),
        checkpoint_info=CheckpointInfo(checkpoint_interval=10, checkpoint_path=checkpoint_path),
        bytes_parser=PythonString(value="def bytes_parser_function(x):\n\treturn x"),
        transform_list=[],
        use_pretrained_model=False,
        pretrained_model_id=-1,
        lr_scheduler=JsonString(value=json.dumps({})),
        grad_scaler_configuration=JsonString(value=json.dumps({})),
        generative=False,
    )


@patch("modyn.trainer_server.internal.utils.training_info.hasattr", return_value=True)
@patch(
    "modyn.trainer_server.internal.utils.training_info.getattr",
    return_value=DummyModelWrapper,
)
def get_training_info(
    training_id,
    temp,
    final_temp,
    storage_address,
    selector_address,
    test_getattr=None,
    test_hasattr=None,
):
    request = get_start_training_request(temp)
    offline_dataset_path = "/tmp/offline_dataset"
    training_info = TrainingInfo(
        request,
        training_id,
        "model",
        json.dumps({}),
        True,
        storage_address,
        selector_address,
        offline_dataset_path,
        pathlib.Path(final_temp),
        pathlib.Path(final_temp) / "log.log",
    )
    return training_info


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_init(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        assert trainer_server._storage_address == "storage:5002"
        assert trainer_server._selector_address == "selector:5003"
        test_connect_to_model_storage.assert_called_with("model_storage:5004")


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_trainer_available(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        response = trainer_server.trainer_available(trainer_available_request, None)
        assert response.available


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch(
    "modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer.hasattr",
    return_value=False,
)
def test_start_training_invalid(test_hasattr, test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        response = trainer_server.start_training(get_start_training_request(), None)
        assert not response.training_started
        assert not trainer_server._training_dict
        assert trainer_server._next_training_id == 0


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch(
    "modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer.hasattr",
    return_value=True,
)
def test_start_training_invalid_id(test_hasattr, test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        req = get_start_training_request()
        req.use_pretrained_model = True
        req.pretrained_model_id = 15
        resp = trainer_server.start_training(req, None)
        assert not resp.training_started


@patch(
    "modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer.download_trained_model",
    return_value=pathlib.Path("downloaded_model.modyn"),
)
@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch(
    "modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer.hasattr",
    return_value=True,
)
@patch(
    "modyn.trainer_server.internal.utils.training_info.getattr",
    return_value=DummyModelWrapper,
)
def test_start_training(
    test_getattr,
    test_hasattr,
    test_connect_to_model_storage,
    download_model_mock: MagicMock,
):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        mock_start = mock.Mock()
        mock_start.side_effect = noop
        trainer_server._training_dict[1] = None
        with patch("multiprocessing.Process.start", mock_start):
            trainer_server.start_training(get_start_training_request(), None)
            assert 0 in trainer_server._training_process_dict
            assert trainer_server._next_training_id == 1

            # start new training
            trainer_server.start_training(get_start_training_request(), None)
            assert 1 in trainer_server._training_process_dict
            assert trainer_server._next_training_id == 2
            assert trainer_server._training_dict[1].model_class_name == "model"
            assert trainer_server._training_dict[1].model_configuration_dict == {}
            assert trainer_server._training_dict[1].amp

            request = get_start_training_request()
            request.use_pretrained_model = True
            request.pretrained_model_id = 10

            resp = trainer_server.start_training(request, None)

            download_model_mock.assert_called_once()
            kwargs = download_model_mock.call_args.kwargs
            remote_file_path = kwargs["remote_path"]
            base_directory = kwargs["base_directory"]
            identifier = kwargs["identifier"]

            assert resp.training_id == 2
            assert str(remote_file_path) == "testpath.modyn"
            assert base_directory == trainer_server._modyn_base_dir
            assert resp.training_started
            assert resp.training_id == identifier
            assert (
                str(trainer_server._training_dict[resp.training_id].pretrained_model_path) == "downloaded_model.modyn"
            )


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_get_training_status_not_registered(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        response = trainer_server.get_training_status(get_status_request, None)
        assert not response.valid


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(TrainerServerGRPCServicer, "get_status_training", return_value=(10, 100, True))
@patch.object(TrainerServerGRPCServicer, "check_for_training_exception")
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint")
def test_get_training_status_alive(
    test_get_latest_checkpoint,
    test_check_for_training_exception,
    test_get_status,
    test_is_alive,
    test_connect_to_model_storage,
):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        trainer_server._training_process_dict[1] = get_training_process_info()
        trainer_server._training_dict[1] = None

        response = trainer_server.get_training_status(get_status_request, None)
        assert response.valid
        assert response.is_running
        assert not response.blocked
        assert response.state_available
        assert response.batches_seen == 10
        assert response.samples_seen == 100
        test_get_latest_checkpoint.assert_not_called()
        test_check_for_training_exception.assert_not_called()


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(TrainerServerGRPCServicer, "get_status_training", return_value=(None, None, None))
@patch.object(TrainerServerGRPCServicer, "check_for_training_exception")
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint")
def test_get_training_status_alive_blocked(
    test_get_latest_checkpoint,
    test_check_for_training_exception,
    test_get_status,
    test_is_alive,
    test_connect_to_model_storage,
):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        trainer_server._training_process_dict[1] = get_training_process_info()
        trainer_server._training_dict[1] = None

        response = trainer_server.get_training_status(get_status_request, None)
        assert response.valid
        assert response.is_running
        assert response.blocked
        assert not response.state_available
        test_get_latest_checkpoint.assert_not_called()
        test_check_for_training_exception.assert_not_called()


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch.object(mp.Process, "is_alive", return_value=False)
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint", return_value=(b"state", 10, 100))
@patch.object(TrainerServerGRPCServicer, "check_for_training_exception", return_value="exception")
@patch.object(TrainerServerGRPCServicer, "get_status_training")
def test_get_training_status_finished_with_exception(
    test_get_status,
    test_check_for_training_exception,
    test_get_latest_checkpoint,
    test_is_alive,
    test_connect_to_model_storage,
):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        trainer_server._training_process_dict[1] = get_training_process_info()
        trainer_server._training_dict[1] = None

        response = trainer_server.get_training_status(get_status_request, None)
        assert response.valid
        assert not response.is_running
        assert not response.blocked
        assert response.state_available
        assert response.batches_seen == 10
        assert response.samples_seen == 100
        assert response.exception == "exception"
        test_get_status.assert_not_called()


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch.object(mp.Process, "is_alive", return_value=False)
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint", return_value=(None, None, None))
@patch.object(TrainerServerGRPCServicer, "check_for_training_exception", return_value="exception")
@patch.object(TrainerServerGRPCServicer, "get_status_training")
def test_get_training_status_finished_no_checkpoint(
    test_get_status,
    test_check_for_training_exception,
    test_get_latest_checkpoint,
    test_is_alive,
    test_connect_to_model_storage,
):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        trainer_server._training_process_dict[1] = get_training_process_info()
        trainer_server._training_dict[1] = None

        response = trainer_server.get_training_status(get_status_request, None)
        assert response.valid
        assert not response.is_running
        assert not response.state_available
        assert response.exception == "exception"
        test_get_status.assert_not_called()


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_get_training_status(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        state_dict = {
            "state": {},
            "num_batches": 10,
            "num_samples": 100,
            "training_active": True,
        }

        training_process_info = get_training_process_info()
        trainer_server._training_process_dict[1] = training_process_info
        training_process_info.status_response_queue_training.put(state_dict)
        num_batches, num_samples, _ = trainer_server.get_status_training(1)
        assert num_batches == state_dict["num_batches"]
        assert num_samples == state_dict["num_samples"]

        timeout = 5
        elapsed = 0

        while True:
            if not platform.system() == "Darwin":
                if training_process_info.status_query_queue_training.qsize() == 1:
                    break
            else:
                if not training_process_info.status_query_queue_training.empty():
                    break

            sleep(1)
            elapsed += 1

            if elapsed >= timeout:
                raise AssertionError("Did not reach desired queue state after 5 seconds.")

        assert training_process_info.status_response_queue_training.empty()
        query = training_process_info.status_query_queue_training.get()
        assert query == TrainerMessages.STATUS_QUERY_MESSAGE


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_check_for_training_exception_not_found(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        trainer_server._training_process_dict[1] = get_training_process_info()
        child_exception = trainer_server.check_for_training_exception(1)
        assert child_exception is None


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_check_for_training_exception_found(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        training_process_info = get_training_process_info()
        trainer_server._training_process_dict[1] = training_process_info

        exception_msg = "exception"
        training_process_info.exception_queue.put(exception_msg)

        child_exception = trainer_server.check_for_training_exception(1)
        assert child_exception == exception_msg


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_get_latest_checkpoint_not_found(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        with tempfile.TemporaryDirectory() as temp:
            with tempfile.TemporaryDirectory() as final_temp:
                trainer_server._training_dict[1] = get_training_info(
                    1,
                    temp,
                    final_temp,
                    trainer_server._storage_address,
                    trainer_server._selector_address,
                )

        training_state, num_batches, num_samples = trainer_server.get_latest_checkpoint(1)
        assert training_state is None
        assert num_batches is None
        assert num_samples is None


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_get_latest_checkpoint_found(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as temp:
        with tempfile.TemporaryDirectory() as final_temp:
            with tempfile.TemporaryDirectory() as modyn_temp:
                trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)

                training_info = get_training_info(
                    1,
                    temp,
                    final_temp,
                    trainer_server._storage_address,
                    trainer_server._selector_address,
                )
                trainer_server._training_dict[1] = training_info

                dict_to_save = {
                    "state": {"weight": 10},
                    "num_batches": 10,
                    "num_samples": 100,
                }

                checkpoint_file = training_info.checkpoint_path / "checkp"
                torch.save(dict_to_save, checkpoint_file)

                checkpoint, num_batches, num_samples = trainer_server.get_latest_checkpoint(1)
                assert num_batches == 10
                assert num_samples == 100

                dict_to_save.pop("num_batches")
                dict_to_save.pop("num_samples")
                with open(checkpoint, "rb") as file:
                    assert torch.load(BytesIO(file.read()))["state"] == dict_to_save["state"]


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_get_latest_checkpoint_invalid(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as temp:
        with tempfile.TemporaryDirectory() as modyn_temp:
            trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)

            training_info = get_training_info(
                1,
                temp,
                trainer_server._storage_address,
                trainer_server._selector_address,
            )
            trainer_server._training_dict[1] = training_info

            dict_to_save = {"state": {"weight": 10}}
            checkpoint_file = training_info.checkpoint_path / "checkp"
            torch.save(dict_to_save, checkpoint_file)

            checkpoint, num_batches, num_samples = trainer_server.get_latest_checkpoint(1)
            assert checkpoint is None
            assert num_batches is None
            assert num_samples is None


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_store_final_model_not_registered(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        response = trainer_server.store_final_model(store_final_model_request, None)
        assert not response.valid_state


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch.object(mp.Process, "is_alive", return_value=True)
def test_store_final_model_still_running(test_is_alive, test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        trainer_server._training_process_dict[1] = get_training_process_info()
        response = trainer_server.store_final_model(store_final_model_request, None)
        assert not response.valid_state


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch.object(mp.Process, "is_alive", return_value=False)
def test_store_final_model_not_found(test_is_alive, test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as temp:
        with tempfile.TemporaryDirectory() as final_temp:
            with tempfile.TemporaryDirectory() as modyn_temp:
                trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
                trainer_server._training_dict[1] = get_training_info(
                    1,
                    temp,
                    final_temp,
                    trainer_server._storage_address,
                    trainer_server._selector_address,
                )
                trainer_server._training_process_dict[1] = get_training_process_info()
                response = trainer_server.store_final_model(store_final_model_request, None)
                assert not response.valid_state


@patch.object(TrainerServerGRPCServicer, "connect_to_model_storage")
@patch.object(mp.Process, "is_alive", return_value=False)
def test_store_final_model_found(test_is_alive, test_connect_to_model_storage):
    model_storage_mock = MagicMock()
    test_connect_to_model_storage.return_value = model_storage_mock
    model_storage_mock.RegisterModel.return_value = RegisterModelResponse(success=True, model_id=1)
    with tempfile.TemporaryDirectory() as temp:
        with tempfile.TemporaryDirectory() as final_temp:
            base_path = pathlib.Path(final_temp)
            trainer_server = TrainerServerGRPCServicer(modyn_config, base_path)
            training_info = get_training_info(
                1,
                temp,
                final_temp,
                trainer_server._storage_address,
                trainer_server._selector_address,
            )
            dict_to_save = {"state": {"weight": 10}}

            checkpoint_file = base_path / "model_final.modyn"
            torch.save(dict_to_save, checkpoint_file)
            checksum = calculate_checksum(checkpoint_file)

            trainer_server._training_dict[1] = training_info
            trainer_server._training_process_dict[1] = get_training_process_info()
            response: StoreFinalModelResponse = trainer_server.store_final_model(store_final_model_request, None)
            assert response.valid_state
            assert response.model_id == 1

            assert not os.path.isfile(checkpoint_file)

            model_storage_mock.RegisterModel.assert_called_once()
            req: RegisterModelRequest = model_storage_mock.RegisterModel.call_args[0][0]
            assert req.pipeline_id == 1
            assert req.trigger_id == 1
            assert req.hostname == "trainer_server"
            assert req.port == 3001
            assert req.model_path == "model_final.modyn"
            assert req.checksum == checksum


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
def test_get_latest_model_not_registered(test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        response = trainer_server.get_latest_model(get_latest_model_request, None)
        assert not response.valid_state


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(TrainerServerGRPCServicer, "get_model_state", return_value=None)
def test_get_latest_model_alive_not_found(test_get_model_state, test_is_alive, test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        trainer_server._training_dict[1] = None
        trainer_server._training_process_dict[1] = get_training_process_info()
        response = trainer_server.get_latest_model(get_latest_model_request, None)
        assert not response.valid_state


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(TrainerServerGRPCServicer, "get_model_state", return_value=b"state")
def test_get_latest_model_alive_found(test_get_model_state, test_is_alive, test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        trainer_server._training_process_dict[1] = get_training_process_info()
        trainer_server._training_dict[1] = None
        response = trainer_server.get_latest_model(get_latest_model_request, None)
        assert response.valid_state
        with open(pathlib.Path(tempdir) / response.model_path, "rb") as file:
            assert file.read() == b"state"


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch.object(mp.Process, "is_alive", return_value=False)
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint", return_value=(None, None, None))
def test_get_latest_model_finished_not_found(test_get_latest_checkpoint, test_is_alive, test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        trainer_server._training_dict[1] = None
        trainer_server._training_process_dict[1] = get_training_process_info()
        response = trainer_server.get_latest_model(get_latest_model_request, None)
        assert not response.valid_state


@patch.object(
    TrainerServerGRPCServicer,
    "connect_to_model_storage",
    return_value=DummyModelStorageStub(),
)
@patch.object(mp.Process, "is_alive", return_value=False)
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint")
def test_get_latest_model_finished_found(test_get_latest_checkpoint, test_is_alive, test_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as tempdir:
        test_get_latest_checkpoint.return_value = (
            pathlib.Path(tempdir) / "testtesttest",
            10,
            100,
        )
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        trainer_server._training_dict[1] = None
        trainer_server._training_process_dict[1] = get_training_process_info()
        response = trainer_server.get_latest_model(get_latest_model_request, None)
        assert response.valid_state
        assert response.model_path == "testtesttest"
