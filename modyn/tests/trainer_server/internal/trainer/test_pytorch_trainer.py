# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import io
import json
import multiprocessing as mp
import os
import platform
import tempfile
from collections import OrderedDict
from io import BytesIO
from time import sleep
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    CheckpointInfo,
    Data,
    JsonString,
    PythonString,
    StartTrainingRequest,
)
from modyn.trainer_server.internal.trainer.pytorch_trainer import PytorchTrainer, train
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo


class MockModule:
    def __init__(self) -> None:
        self.model = MockModelWrapper

    def train(self) -> None:
        pass


class MockModelWrapper:
    def __init__(self, model_configuration=None) -> None:
        self.model = MockModel()


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, data):
        return data


class MockDataset(torch.utils.data.IterableDataset):
    # pylint: disable=abstract-method, useless-parent-delegation
    def __init__(self) -> None:
        super().__init__()

    def __iter__(self):
        return iter(range(100))


def get_mock_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn x"


def mock_get_dataloaders(training_id, dataset_id, num_dataloaders, batch_size, bytes_parser, transform_list, sample_id):
    mock_train_dataloader = iter(
        [(torch.ones(8, 10, requires_grad=True), torch.ones(8, dtype=int)) for _ in range(100)]
    )
    return mock_train_dataloader, None


@patch("modyn.trainer_server.internal.utils.training_info.dynamic_module_import")
def get_training_info(use_pretrained: bool, pretrained_model: Any, dynamic_module_patch: MagicMock):
    with tempfile.TemporaryDirectory() as tmpdirname:
        dynamic_module_patch.return_value = MockModule()
        request = StartTrainingRequest(
            pipeline_id=1,
            trigger_id=1,
            device="cpu",
            data_info=Data(dataset_id="MNIST", num_dataloaders=2),
            optimizer_parameters=JsonString(value=json.dumps({"lr": 0.1})),
            model_configuration=JsonString(value=json.dumps({})),
            criterion_parameters=JsonString(value=json.dumps({})),
            model_id="model",
            torch_optimizer="SGD",
            batch_size=32,
            torch_criterion="CrossEntropyLoss",
            checkpoint_info=CheckpointInfo(checkpoint_interval=10, checkpoint_path=tmpdirname),
            bytes_parser=PythonString(value=get_mock_bytes_parser()),
            transform_list=[],
            use_pretrained_model=use_pretrained,
            pretrained_model=pretrained_model
        )
        training_info = TrainingInfo(request)
        return training_info


@patch("modyn.trainer_server.internal.utils.training_info.dynamic_module_import")
def get_mock_trainer(query_queue: mp.Queue(), response_queue: mp.Queue(), use_pretrained: bool, pretrained_model: Any, dynamic_module_patch: MagicMock):
    dynamic_module_patch.return_value = MockModule()
    training_info = get_training_info(use_pretrained, pretrained_model)
    trainer = PytorchTrainer(training_info, "cpu", query_queue, response_queue)
    return trainer


def test_trainer_init():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), False, None)
    assert isinstance(trainer._model, MockModelWrapper)
    assert isinstance(trainer._optimizer, torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)


@patch.object(PytorchTrainer, "load_state_if_given")
def test_trainer_init_from_pretrained_model(test_load_state_if_given):
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), True, b"state")
    assert isinstance(trainer._model, MockModelWrapper)
    assert isinstance(trainer._optimizer, torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    test_load_state_if_given.assert_called_once_with(b"state")


def test_save_state_to_file():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), False, None)
    with tempfile.NamedTemporaryFile() as temp:
        trainer.save_state(temp.name, 10)
        assert os.path.exists(temp.name)
        saved_dict = torch.load(temp.name)

    assert saved_dict == {
        "model": OrderedDict([("_weight", torch.tensor([1.0]))]),
        "optimizer": {
            "state": {},
            "param_groups": [
                {
                    "lr": 0.1,
                    "momentum": 0,
                    "dampening": 0,
                    "weight_decay": 0,
                    "nesterov": False,
                    "maximize": False,
                    "foreach": None,
                    "differentiable": False,
                    "params": [0],
                }
            ],
        },
        "iteration": 10,
    }


def test_save_state_to_buffer():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), False, None)
    buffer = io.BytesIO()
    trainer.save_state(buffer)
    buffer.seek(0)
    saved_dict = torch.load(buffer)
    assert saved_dict == {
        "model": OrderedDict([("_weight", torch.tensor([1.0]))]),
        "optimizer": {
            "state": {},
            "param_groups": [
                {
                    "lr": 0.1,
                    "momentum": 0,
                    "dampening": 0,
                    "weight_decay": 0,
                    "nesterov": False,
                    "maximize": False,
                    "foreach": None,
                    "differentiable": False,
                    "params": [0],
                }
            ],
        },
    }


def test_load_state_if_given():
    dict_to_save = {
        "model": OrderedDict([("_weight", torch.tensor([100.0]))]),
        "optimizer": {
            "state": {},
            "param_groups": [
                {
                    "lr": 0.1,
                    "momentum": 0,
                    "dampening": 0,
                    "weight_decay": 0,
                    "nesterov": False,
                    "maximize": False,
                    "foreach": None,
                    "differentiable": False,
                    "params": [0],
                }
            ],
        },
    }
    initial_state_buffer = io.BytesIO()
    torch.save(dict_to_save, initial_state_buffer)
    initial_state_buffer.seek(0)
    initial_state = initial_state_buffer.read()
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), True, initial_state)
    assert trainer._model.model.state_dict() == dict_to_save["model"]
    assert trainer._optimizer.state_dict() == dict_to_save["optimizer"]


# def test_load_checkpoint():
#     trainer = get_mock_trainer(mp.Queue(), mp.Queue())

#     dict_to_save = {
#         "model": OrderedDict([("_weight", torch.tensor([100.0]))]),
#         "optimizer": {
#             "state": {},
#             "param_groups": [
#                 {
#                     "lr": 0.1,
#                     "momentum": 0,
#                     "dampening": 0,
#                     "weight_decay": 0,
#                     "nesterov": False,
#                     "maximize": False,
#                     "foreach": None,
#                     "differentiable": False,
#                     "params": [0],
#                 }
#             ],
#         },
#         "iteration": 100,
#     }

#     with tempfile.NamedTemporaryFile() as temp:
#         torch.save(dict_to_save, temp.name)
#         trainer.load_checkpoint(temp.name)
#         assert trainer._model.model.state_dict() == dict_to_save["model"]
#         assert trainer._optimizer.state_dict() == dict_to_save["optimizer"]


def test_send_state_to_server():
    response_queue = mp.Queue()
    query_queue = mp.Queue()
    trainer = get_mock_trainer(query_queue, response_queue, False, None)
    trainer.send_state_to_server(20)
    response = response_queue.get()
    assert response["num_batches"] == 20
    file_like = BytesIO(response["state"])
    assert torch.load(file_like) == {
        "model": OrderedDict([("_weight", torch.tensor([1.0]))]),
        "optimizer": {
            "state": {},
            "param_groups": [
                {
                    "lr": 0.1,
                    "momentum": 0,
                    "dampening": 0,
                    "weight_decay": 0,
                    "nesterov": False,
                    "maximize": False,
                    "foreach": None,
                    "differentiable": False,
                    "params": [0],
                }
            ],
        },
    }


@patch(
    "modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders",
    mock_get_dataloaders,
)
def test_train_invalid_query_message():
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    trainer = get_mock_trainer(query_status_queue, status_queue, False, None)
    query_status_queue.put("INVALID MESSAGE")
    timeout = 5
    elapsed = 0
    while query_status_queue.empty():
        sleep(1)
        elapsed += 1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    with tempfile.NamedTemporaryFile() as temp:
        with pytest.raises(ValueError, match="Unknown message in the status query queue"):
            trainer.train(temp.name)

        elapsed = 0
        while not (query_status_queue.empty() and status_queue.empty()):
            sleep(1)
            elapsed += 1

            if elapsed >= timeout:
                raise TimeoutError("Did not reach desired queue state within timelimit.")


@patch(
    "modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders",
    mock_get_dataloaders,
)
def test_train():
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    trainer = get_mock_trainer(query_status_queue, status_queue, False, None)
    query_status_queue.put(TrainerMessages.STATUS_QUERY_MESSAGE)
    timeout = 5
    elapsed = 0
    while query_status_queue.empty():
        sleep(1)
        elapsed += 1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    with tempfile.NamedTemporaryFile() as temp:
        trainer.train(temp.name)
        assert os.path.exists(temp.name)
        assert trainer._num_samples == 800
        while not query_status_queue.empty():
            sleep(1)
            elapsed += 1

            if elapsed >= timeout:
                raise TimeoutError("Did not reach desired queue state within timelimit.")

        elapsed = 0
        while True:
            if not platform.system() == "Darwin":
                if status_queue.qsize() == 1:
                    break
            else:
                if not status_queue.empty():
                    break

            sleep(1)
            elapsed += 1

            if elapsed >= timeout:
                raise AssertionError("Did not reach desired queue state after 5 seconds.")

        status = status_queue.get()
        assert status["num_batches"] == 0
        assert status["num_samples"] == 0
        status_state = torch.load(io.BytesIO(status["state"]))
        assert status_state == {
            "model": OrderedDict([("_weight", torch.tensor([1.0]))]),
            "optimizer": {
                "state": {},
                "param_groups": [
                    {
                        "lr": 0.1,
                        "momentum": 0,
                        "dampening": 0,
                        "weight_decay": 0,
                        "nesterov": False,
                        "maximize": False,
                        "foreach": None,
                        "differentiable": False,
                        "params": [0],
                    }
                ],
            },
        }
        assert os.path.exists(trainer._checkpoint_path + "/model_final.pt")


@patch("modyn.trainer_server.internal.utils.training_info.dynamic_module_import")
@patch(
    "modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders",
    mock_get_dataloaders,
)
def test_create_trainer_with_exception(test_dynamic_module_import):
    test_dynamic_module_import.return_value = MockModule()
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    exception_queue = mp.Queue()
    training_info = get_training_info(False, None)
    query_status_queue.put("INVALID MESSAGE")
    timeout = 5
    elapsed = 0
    while query_status_queue.empty():
        sleep(1)
        elapsed += 1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    with tempfile.NamedTemporaryFile() as temp:
        train(
            training_info,
            "cpu",
            temp.name,
            exception_queue,
            query_status_queue,
            status_queue,
        )
        elapsed = 0
        while not (query_status_queue.empty() and status_queue.empty()):
            sleep(1)
            elapsed += 1

            if elapsed >= timeout:
                raise TimeoutError("Did not reach desired queue state within timelimit.")

        elapsed = 0

        while True:
            if not platform.system() == "Darwin":
                if exception_queue.qsize() == 1:
                    break
            else:
                if not exception_queue.empty():
                    break

            sleep(1)
            elapsed += 1

            if elapsed >= timeout:
                raise AssertionError("Did not reach desired queue state after 5 seconds.")

        exception = exception_queue.get()
        assert "ValueError: Unknown message in the status query queue" in exception
