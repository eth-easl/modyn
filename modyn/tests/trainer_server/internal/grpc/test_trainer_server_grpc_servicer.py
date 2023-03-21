# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import json
import multiprocessing as mp
import pathlib
import platform
import tempfile
from io import BytesIO
from time import sleep
from unittest import mock
from unittest.mock import patch

import torch
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    CheckpointInfo,
    Data,
    GetFinalModelRequest,
    GetLatestModelRequest,
    JsonString,
    PythonString,
    StartTrainingRequest,
    TrainerAvailableRequest,
    TrainingStatusRequest,
)
from modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer import TrainerServerGRPCServicer
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo
from modyn.trainer_server.internal.utils.training_process_info import TrainingProcessInfo

trainer_available_request = TrainerAvailableRequest()
get_status_request = TrainingStatusRequest(training_id=1)
get_final_model_request = GetFinalModelRequest(training_id=1)
get_latest_model_request = GetLatestModelRequest(training_id=1)

modyn_config = {
    "trainer_server": {"hostname": "trainer_server", "port": "5001"},
    "storage": {"hostname": "storage", "port": "5002"},
    "selector": {"hostname": "selector", "port": "5003"},
}


class DummyModelWrapper:
    def __init__(self, model_configuration=None) -> None:
        self.model = None


def noop():
    return


def get_training_process_info():
    status_query_queue = mp.Queue()
    status_response_queue = mp.Queue()
    exception_queue = mp.Queue()

    training_process_info = TrainingProcessInfo(
        mp.Process(), exception_queue, status_query_queue, status_response_queue
    )
    return training_process_info


def get_start_training_request(checkpoint_path="", valid_model=True):
    return StartTrainingRequest(
        pipeline_id=1,
        trigger_id=1,
        device="cpu",
        amp=False,
        model_id="model" if valid_model else "unknown",
        batch_size=32,
        torch_optimizers_configuration=JsonString(
            value=json.dumps(
                {"default": {"algorithm": "SGD", "param_groups": [{"module": "model", "config": {"lr": 0.1}}]}}
            )
        ),
        torch_criterion="CrossEntropyLoss",
        model_configuration=JsonString(value=json.dumps({})),
        criterion_parameters=JsonString(value=json.dumps({})),
        data_info=Data(dataset_id="Dataset", num_dataloaders=1),
        checkpoint_info=CheckpointInfo(checkpoint_interval=10, checkpoint_path=checkpoint_path),
        bytes_parser=PythonString(value="def bytes_parser_function(x):\n\treturn x"),
        transform_list=[],
        use_pretrained_model=False,
        pretrained_model_path="",
        lr_scheduler=JsonString(value=json.dumps({})),
        grad_scaler_configuration=JsonString(value=json.dumps({})),
    )


@patch("modyn.trainer_server.internal.utils.training_info.hasattr", return_value=True)
@patch(
    "modyn.trainer_server.internal.utils.training_info.getattr",
    return_value=DummyModelWrapper,
)
def get_training_info(
    training_id, temp, final_temp, storage_address, selector_address, test_getattr=None, test_hasattr=None
):
    request = get_start_training_request(temp)
    training_info = TrainingInfo(request, training_id, storage_address, selector_address, pathlib.Path(final_temp))
    return training_info


def test_init():
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        assert trainer_server._storage_address == "storage:5002"
        assert trainer_server._selector_address == "selector:5003"


def test_trainer_available():
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        response = trainer_server.trainer_available(trainer_available_request, None)
        assert response.available


@patch.object(mp.Process, "is_alive", return_value=True)
def test_trainer_not_available(test_is_alive):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        trainer_server._training_process_dict[10] = TrainingProcessInfo(
            mp.Process(), mp.Queue(), mp.Queue(), mp.Queue()
        )
        response = trainer_server.trainer_available(trainer_available_request, None)
        assert not response.available


@patch("modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer.hasattr", return_value=False)
def test_start_training_invalid(test_hasattr):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        response = trainer_server.start_training(get_start_training_request(valid_model=False), None)
        assert not response.training_started
        assert not trainer_server._training_dict
        assert trainer_server._next_training_id == 0


@patch("modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer.hasattr", return_value=True)
@patch(
    "modyn.trainer_server.internal.utils.training_info.getattr",
    return_value=DummyModelWrapper,
)
def test_start_training(test_getattr, test_hasattr):
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        mock_start = mock.Mock()
        mock_start.side_effect = noop
        trainer_server._training_dict[1] = None
        with patch("multiprocessing.Process.start", mock_start):
            trainer_server.start_training(get_start_training_request(valid_model=True), None)
            assert 0 in trainer_server._training_process_dict
            assert trainer_server._next_training_id == 1

            # start new training
            trainer_server.start_training(get_start_training_request(valid_model=True), None)
            assert 1 in trainer_server._training_process_dict
            assert trainer_server._next_training_id == 2


def test_get_training_status_not_registered():
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        response = trainer_server.get_training_status(get_status_request, None)
        assert not response.valid


@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(TrainerServerGRPCServicer, "get_status", return_value=(10, 100))
@patch.object(TrainerServerGRPCServicer, "check_for_training_exception")
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint")
def test_get_training_status_alive(
    test_get_latest_checkpoint,
    test_check_for_training_exception,
    test_get_status,
    test_is_alive,
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


@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(TrainerServerGRPCServicer, "get_status", return_value=(None, None))
@patch.object(TrainerServerGRPCServicer, "check_for_training_exception")
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint")
def test_get_training_status_alive_blocked(
    test_get_latest_checkpoint,
    test_check_for_training_exception,
    test_get_status,
    test_is_alive,
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


@patch.object(mp.Process, "is_alive", return_value=False)
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint", return_value=(b"state", 10, 100))
@patch.object(TrainerServerGRPCServicer, "check_for_training_exception", return_value="exception")
@patch.object(TrainerServerGRPCServicer, "get_status")
def test_get_training_status_finished_with_exception(
    test_get_status,
    test_check_for_training_exception,
    test_get_latest_checkpoint,
    test_is_alive,
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


@patch.object(mp.Process, "is_alive", return_value=False)
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint", return_value=(None, None, None))
@patch.object(TrainerServerGRPCServicer, "check_for_training_exception", return_value="exception")
@patch.object(TrainerServerGRPCServicer, "get_status")
def test_get_training_status_finished_no_checkpoint(
    test_get_status,
    test_check_for_training_exception,
    test_get_latest_checkpoint,
    test_is_alive,
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


def test_get_training_status():
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        state_dict = {"state": {}, "num_batches": 10, "num_samples": 100}

        training_process_info = get_training_process_info()
        trainer_server._training_process_dict[1] = training_process_info
        training_process_info.status_response_queue.put(state_dict)
        num_batches, num_samples = trainer_server.get_status(1)
        assert num_batches == state_dict["num_batches"]
        assert num_samples == state_dict["num_samples"]

        timeout = 5
        elapsed = 0

        while True:
            if not platform.system() == "Darwin":
                if training_process_info.status_query_queue.qsize() == 1:
                    break
            else:
                if not training_process_info.status_query_queue.empty():
                    break

            sleep(1)
            elapsed += 1

            if elapsed >= timeout:
                raise AssertionError("Did not reach desired queue state after 5 seconds.")

        assert training_process_info.status_response_queue.empty()
        query = training_process_info.status_query_queue.get()
        assert query == TrainerMessages.STATUS_QUERY_MESSAGE


def test_check_for_training_exception_not_found():
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        trainer_server._training_process_dict[1] = get_training_process_info()
        child_exception = trainer_server.check_for_training_exception(1)
        assert child_exception is None


def test_check_for_training_exception_found():
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        training_process_info = get_training_process_info()
        trainer_server._training_process_dict[1] = training_process_info

        exception_msg = "exception"
        training_process_info.exception_queue.put(exception_msg)

        child_exception = trainer_server.check_for_training_exception(1)
        assert child_exception == exception_msg


def test_get_latest_checkpoint_not_found():
    with tempfile.TemporaryDirectory() as modyn_temp:
        trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
        with tempfile.TemporaryDirectory() as temp:
            with tempfile.TemporaryDirectory() as final_temp:
                trainer_server._training_dict[1] = get_training_info(
                    1, temp, final_temp, trainer_server._storage_address, trainer_server._selector_address
                )

        training_state, num_batches, num_samples = trainer_server.get_latest_checkpoint(1)
        assert training_state is None
        assert num_batches is None
        assert num_samples is None


def test_get_latest_checkpoint_found():
    with tempfile.TemporaryDirectory() as temp:
        with tempfile.TemporaryDirectory() as final_temp:
            with tempfile.TemporaryDirectory() as modyn_temp:
                trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)

                training_info = get_training_info(
                    1, temp, final_temp, trainer_server._storage_address, trainer_server._selector_address
                )
                trainer_server._training_dict[1] = training_info

                dict_to_save = {"state": {"weight": 10}, "num_batches": 10, "num_samples": 100}

                checkpoint_file = training_info.checkpoint_path / "checkp"
                torch.save(dict_to_save, checkpoint_file)

                checkpoint, num_batches, num_samples = trainer_server.get_latest_checkpoint(1)
                assert num_batches == 10
                assert num_samples == 100

                dict_to_save.pop("num_batches")
                dict_to_save.pop("num_samples")
                with open(checkpoint, "rb") as file:
                    assert torch.load(BytesIO(file.read()))["state"] == dict_to_save["state"]


def test_get_latest_checkpoint_invalid():
    with tempfile.TemporaryDirectory() as temp:
        with tempfile.TemporaryDirectory() as modyn_temp:
            trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)

            training_info = get_training_info(
                1, temp, trainer_server._storage_address, trainer_server._selector_address
            )
            trainer_server._training_dict[1] = training_info

            dict_to_save = {"state": {"weight": 10}}
            checkpoint_file = training_info.checkpoint_path / "checkp"
            torch.save(dict_to_save, checkpoint_file)

            checkpoint, num_batches, num_samples = trainer_server.get_latest_checkpoint(1)
            assert checkpoint is None
            assert num_batches is None
            assert num_samples is None


def test_get_final_model_not_registered():
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        response = trainer_server.get_final_model(get_final_model_request, None)
        assert not response.valid_state


@patch.object(mp.Process, "is_alive", return_value=True)
def test_get_final_model_still_running(test_is_alive):
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        trainer_server._training_process_dict[1] = get_training_process_info()
        response = trainer_server.get_final_model(get_final_model_request, None)
        assert not response.valid_state


@patch.object(mp.Process, "is_alive", return_value=False)
def test_get_final_model_not_found(test_is_alive):
    with tempfile.TemporaryDirectory() as temp:
        with tempfile.TemporaryDirectory() as final_temp:
            with tempfile.TemporaryDirectory() as modyn_temp:
                trainer_server = TrainerServerGRPCServicer(modyn_config, modyn_temp)
                trainer_server._training_dict[1] = get_training_info(
                    1, temp, final_temp, trainer_server._storage_address, trainer_server._selector_address
                )
                trainer_server._training_process_dict[1] = get_training_process_info()
                response = trainer_server.get_final_model(get_final_model_request, None)
                assert not response.valid_state


@patch.object(mp.Process, "is_alive", return_value=False)
def test_get_final_model_found(test_is_alive):
    with tempfile.TemporaryDirectory() as temp:
        with tempfile.TemporaryDirectory() as final_temp:
            base_path = pathlib.Path(final_temp)
            trainer_server = TrainerServerGRPCServicer(modyn_config, base_path)
            training_info = get_training_info(
                1, temp, final_temp, trainer_server._storage_address, trainer_server._selector_address
            )
            dict_to_save = {"state": {"weight": 10}}

            checkpoint_file = base_path / "model_final.modyn"
            torch.save(dict_to_save, checkpoint_file)

            trainer_server._training_dict[1] = training_info
            trainer_server._training_process_dict[1] = get_training_process_info()
            response = trainer_server.get_final_model(get_final_model_request, None)
            assert response.valid_state
            with open(base_path / response.model_path, "rb") as file:
                assert torch.load(BytesIO(file.read())) == {"state": {"weight": 10}}


def test_get_latest_model_not_registered():
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        response = trainer_server.get_latest_model(get_latest_model_request, None)
        assert not response.valid_state


@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(TrainerServerGRPCServicer, "get_model_state", return_value=None)
def test_get_latest_model_alive_not_found(test_get_model_state, test_is_alive):
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        trainer_server._training_dict[1] = None
        trainer_server._training_process_dict[1] = get_training_process_info()
        response = trainer_server.get_latest_model(get_latest_model_request, None)
        assert not response.valid_state


@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(TrainerServerGRPCServicer, "get_model_state", return_value=b"state")
def test_get_latest_model_alive_found(test_get_model_state, test_is_alive):
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        trainer_server._training_process_dict[1] = get_training_process_info()
        trainer_server._training_dict[1] = None
        response = trainer_server.get_latest_model(get_latest_model_request, None)
        assert response.valid_state
        with open(pathlib.Path(tempdir) / response.model_path, "rb") as file:
            assert file.read() == b"state"


@patch.object(mp.Process, "is_alive", return_value=False)
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint", return_value=(None, None, None))
def test_get_latest_model_finished_not_found(test_get_latest_checkpoint, test_is_alive):
    with tempfile.TemporaryDirectory() as tempdir:
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        trainer_server._training_dict[1] = None
        trainer_server._training_process_dict[1] = get_training_process_info()
        response = trainer_server.get_latest_model(get_latest_model_request, None)
        assert not response.valid_state


@patch.object(mp.Process, "is_alive", return_value=False)
@patch.object(TrainerServerGRPCServicer, "get_latest_checkpoint")
def test_get_latest_model_finished_found(test_get_latest_checkpoint, test_is_alive):
    with tempfile.TemporaryDirectory() as tempdir:
        test_get_latest_checkpoint.return_value = (pathlib.Path(tempdir) / "testtesttest", 10, 100)
        trainer_server = TrainerServerGRPCServicer(modyn_config, tempdir)
        trainer_server._training_dict[1] = None
        trainer_server._training_process_dict[1] = get_training_process_info()
        response = trainer_server.get_latest_model(get_latest_model_request, None)
        assert response.valid_state
        assert response.model_path == "testtesttest"
