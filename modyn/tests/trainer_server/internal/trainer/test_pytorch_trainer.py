# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import io
import json
import logging
import multiprocessing as mp
import os
import pathlib
import platform
import tempfile
from collections import OrderedDict
from io import BytesIO
from time import sleep
from unittest.mock import MagicMock, patch

import grpc
import pytest
import torch
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    CheckpointInfo,
    Data,
    JsonString,
    PythonString,
    StartTrainingRequest,
)
from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector
from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.base_callback import BaseCallback
from modyn.trainer_server.internal.trainer.pytorch_trainer import PytorchTrainer, train
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo


class NoneOrFalse:
    def __eq__(self, other):
        if other is None or not other:
            return True

        return False


class MockModule:
    def __init__(self, num_optimizers) -> None:
        if num_optimizers == 1:
            self.model = MockModelWrapper
        else:
            self.model = MockSuperModelWrapper

    def train(self) -> None:
        pass


class MockModelWrapper:
    def __init__(self, model_configuration=None, device="cpu", amp=False) -> None:
        self.model = MockModel()


class MockSuperModelWrapper:
    def __init__(self, model_configuration=None, device="cpu", amp=False) -> None:
        self.model = MockSuperModel()


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, data):
        return data


class MockSuperModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moda = MockModel()
        self.modb = MockModel()
        self.modc = MockModel()

    def forward(self, data):
        return self.moda(self.modb(data))


class MockDataset(torch.utils.data.IterableDataset):
    # pylint: disable=abstract-method, useless-parent-delegation
    def __init__(self) -> None:
        super().__init__()

    def __iter__(self):
        return iter(range(100))


class MockLRSchedulerModule:
    def __init__(self) -> None:
        self.custom_scheduler = CustomLRScheduler


# pylint: disable=dangerous-default-value
class CustomLRScheduler:
    def __init__(self, optimizers, config={}) -> None:
        pass

    def step(self):
        pass


def get_mock_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn x"


def get_mock_label_transformer():
    return (
        "import torch\ndef label_transformer_function(x: torch.Tensor) -> "
        "torch.Tensor:\n\treturn x.to(torch.float32)"
    )


def mock_get_dataloaders(
    pipeline_id,
    trigger_id,
    dataset_id,
    num_dataloaders,
    batch_size,
    bytes_parser,
    transform,
    storage_address,
    selector_address,
    training_id,
    return_weights,
):
    mock_train_dataloader = iter(
        [(("1",) * 8, torch.ones(8, 10, requires_grad=True), torch.ones(8, dtype=int)) for _ in range(100)]
    )
    return mock_train_dataloader, None


def noop_constructor_mock(self, channel):
    pass


# # pylint: disable=too-many-locals
@patch("modyn.trainer_server.internal.utils.training_info.dynamic_module_import")
def get_training_info(
    training_id: int,
    use_pretrained: bool,
    load_optimizer_state: bool,
    pretrained_model_path: pathlib.Path,
    storage_address: str,
    selector_address: str,
    num_optimizers: int,
    lr_scheduler: str,
    transform_label: bool,
    model_dynamic_module_patch: MagicMock,
):
    if num_optimizers == 1:
        torch_optimizers_configuration = {
            "default": {
                "algorithm": "SGD",
                "source": "PyTorch",
                "param_groups": [{"module": "model", "config": {"lr": 0.1}}],
            }
        }
    else:
        torch_optimizers_configuration = {
            "opt1": {
                "algorithm": "SGD",
                "source": "PyTorch",
                "param_groups": [{"module": "model.moda", "config": {"lr": 0.1}}],
            },
            "opt2": {
                "algorithm": "Adam",
                "source": "PyTorch",
                "param_groups": [
                    {"module": "model.modb", "config": {"lr": 0.5}},
                    {"module": "model.modc", "config": {"lr": 0.8}},
                ],
            },
        }

    if lr_scheduler == "torch":
        lr_scheduler_config = {
            "name": "StepLR",
            "source": "PyTorch",
            "optimizers": ["default"] if num_optimizers == 1 else ["opt1"],
            "config": {"step_size": 10},
        }
    elif lr_scheduler == "custom":
        lr_scheduler_config = {
            "name": "custom_scheduler",
            "source": "Custom",
            "optimizers": ["default"] if num_optimizers == 1 else ["opt1", "opt2"],
            "config": {},
        }
    else:
        lr_scheduler_config = {}

    with tempfile.TemporaryDirectory() as tmpdirname:
        with tempfile.TemporaryDirectory() as final_tmpdirname:
            model_dynamic_module_patch.return_value = MockModule(num_optimizers)
            request = StartTrainingRequest(
                pipeline_id=1,
                trigger_id=1,
                device="cpu",
                amp=False,
                data_info=Data(dataset_id="MNIST", num_dataloaders=2),
                torch_optimizers_configuration=JsonString(value=json.dumps(torch_optimizers_configuration)),
                model_configuration=JsonString(value=json.dumps({})),
                criterion_parameters=JsonString(value=json.dumps({})),
                model_id="model",
                batch_size=32,
                torch_criterion="CrossEntropyLoss",
                checkpoint_info=CheckpointInfo(checkpoint_interval=10, checkpoint_path=tmpdirname),
                bytes_parser=PythonString(value=get_mock_bytes_parser()),
                transform_list=[],
                use_pretrained_model=use_pretrained,
                load_optimizer_state=load_optimizer_state,
                pretrained_model_path=str(pretrained_model_path),
                lr_scheduler=JsonString(value=json.dumps(lr_scheduler_config)),
                label_transformer=PythonString(value=get_mock_label_transformer() if transform_label else ""),
                grad_scaler_configuration=JsonString(value=json.dumps({})),
                epochs_per_trigger=1,
            )
            training_info = TrainingInfo(
                request,
                training_id,
                storage_address,
                selector_address,
                pathlib.Path(final_tmpdirname),
                pretrained_model_path,
            )
            return training_info


@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch("modyn.trainer_server.internal.utils.training_info.dynamic_module_import")
@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.dynamic_module_import")
@patch.object(PytorchTrainer, "connect_to_selector", return_value=None)
@patch.object(PytorchTrainer, "get_selection_strategy", return_value=(False, "", {}))
def get_mock_trainer(
    query_queue: mp.Queue(),
    response_queue: mp.Queue(),
    use_pretrained: bool,
    load_optimizer_state: bool,
    pretrained_model_path: pathlib.Path,
    num_optimizers: int,
    lr_scheduler: str,
    transform_label: bool,
    mock_selection_strategy: MagicMock,
    mock_selector_connection: MagicMock,
    lr_scheduler_dynamic_module_patch: MagicMock,
    model_dynamic_module_patch: MagicMock,
    test_insecure_channel: MagicMock,
    test_grpc_connection_established: MagicMock,
):
    model_dynamic_module_patch.return_value = MockModule(num_optimizers)
    lr_scheduler_dynamic_module_patch.return_value = MockLRSchedulerModule()
    training_info = get_training_info(
        0,
        use_pretrained,
        load_optimizer_state,
        pretrained_model_path,
        "",
        "",
        num_optimizers,
        lr_scheduler,
        transform_label,
    )
    trainer = PytorchTrainer(training_info, "cpu", query_queue, response_queue, logging.getLogger(__name__))
    return trainer


def test_trainer_init():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), False, False, None, 1, "", False)
    assert isinstance(trainer._model, MockModelWrapper)
    assert len(trainer._optimizers) == 1
    assert isinstance(trainer._optimizers["default"], torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert not trainer._lr_scheduler
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    assert trainer._label_tranformer_function is None


def test_trainer_init_multi_optimizers():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), False, False, b"state", 2, "", False)
    assert isinstance(trainer._model, MockSuperModelWrapper)
    assert len(trainer._optimizers) == 2
    assert isinstance(trainer._optimizers["opt1"], torch.optim.SGD)
    assert isinstance(trainer._optimizers["opt2"], torch.optim.Adam)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert not trainer._lr_scheduler
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    assert trainer._label_tranformer_function is None


def test_trainer_init_torch_lr_scheduler():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), False, False, None, 1, "torch", False)
    assert isinstance(trainer._model, MockModelWrapper)
    assert len(trainer._optimizers) == 1
    assert isinstance(trainer._optimizers["default"], torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(trainer._lr_scheduler, torch.optim.lr_scheduler.StepLR)
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    assert trainer._label_tranformer_function is None


def test_trainer_init_custom_lr_scheduler():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), False, False, None, 1, "custom", False)
    assert isinstance(trainer._model, MockModelWrapper)
    assert len(trainer._optimizers) == 1
    assert isinstance(trainer._optimizers["default"], torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(trainer._lr_scheduler, CustomLRScheduler)
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    assert trainer._label_tranformer_function is None


@patch.object(PytorchTrainer, "load_state_if_given")
def test_trainer_init_from_pretrained_model(load_state_if_given_mock):
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), True, False, "/path/to/model", 1, "", False)
    assert isinstance(trainer._model, MockModelWrapper)
    assert isinstance(trainer._optimizers["default"], torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    load_state_if_given_mock.assert_called_once_with("/path/to/model", False)
    assert trainer._label_tranformer_function is None


def test_trainer_init_with_label_transformer():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), False, False, None, 1, "", True)
    assert isinstance(trainer._model, MockModelWrapper)
    assert len(trainer._optimizers) == 1
    assert isinstance(trainer._optimizers["default"], torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert not trainer._lr_scheduler
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    assert trainer._label_tranformer_function is not None
    test_tensor = torch.ones(10, dtype=torch.int32)
    assert torch.equal(trainer._label_tranformer_function(test_tensor), torch.ones(10, dtype=torch.float32))
    assert trainer._label_tranformer_function(test_tensor).dtype == torch.float32


def test_save_state_to_file():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), False, False, None, 2, "", False)
    with tempfile.NamedTemporaryFile() as temp:
        trainer.save_state(temp.name, 10)
        assert os.path.exists(temp.name)
        saved_dict = torch.load(temp.name)

    assert saved_dict == {
        "model": OrderedDict(
            [
                ("moda._weight", torch.tensor([1.0])),
                ("modb._weight", torch.tensor([1.0])),
                ("modc._weight", torch.tensor([1.0])),
            ]
        ),
        "optimizer-opt1": {
            "state": {},
            "param_groups": [
                {
                    "lr": pytest.approx(0.1),
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
        "optimizer-opt2": {
            "state": {},
            "param_groups": [
                {
                    "lr": pytest.approx(0.5),
                    "betas": (pytest.approx(0.9), pytest.approx(0.999)),
                    "eps": pytest.approx(1e-08),
                    "weight_decay": 0,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": NoneOrFalse(),
                    "params": [0],
                },
                {
                    "lr": pytest.approx(0.8),
                    "betas": (pytest.approx(0.9), pytest.approx(0.999)),
                    "eps": pytest.approx(1e-08),
                    "weight_decay": 0,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": NoneOrFalse(),
                    "params": [1],
                },
            ],
        },
        "iteration": 10,
    }


def test_save_state_to_buffer():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue(), False, False, None, 1, "", False)
    buffer = io.BytesIO()
    trainer.save_state(buffer)
    buffer.seek(0)
    saved_dict = torch.load(buffer)
    assert saved_dict == {
        "model": OrderedDict([("_weight", torch.tensor([1.0]))]),
        "optimizer-default": {
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
        "model": OrderedDict(
            [
                ("moda._weight", torch.tensor([1.0])),
                ("modb._weight", torch.tensor([1.0])),
                ("modc._weight", torch.tensor([1.0])),
            ]
        ),
        "optimizer-opt1": {
            "state": {},
            "param_groups": [
                {
                    "lr": 1.0,
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
        "optimizer-opt2": {
            "state": {},
            "param_groups": [
                {
                    "lr": 1.5,
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": False,
                    "params": [0],
                },
                {
                    "lr": 1.8,
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": False,
                    "params": [1],
                },
            ],
        },
    }
    with tempfile.TemporaryDirectory() as tempdir:
        initial_state_buffer = io.BytesIO()
        torch.save(dict_to_save, initial_state_buffer)
        initial_state_buffer.seek(0)
        state_path = pathlib.Path(tempdir) / "test.state"
        with open(state_path, "wb") as file:
            file.write(initial_state_buffer.read())

        trainer = get_mock_trainer(mp.Queue(), mp.Queue(), True, True, state_path, 2, "", False)
        assert trainer._model.model.state_dict() == dict_to_save["model"]
        assert trainer._optimizers["opt1"].state_dict() == dict_to_save["optimizer-opt1"]
        assert trainer._optimizers["opt2"].state_dict() == dict_to_save["optimizer-opt2"]
        initial_state_buffer.seek(0)
        state_path = pathlib.Path(tempdir) / "test.state"
        with open(state_path, "wb") as file:
            file.write(initial_state_buffer.read())

        new_trainer = get_mock_trainer(mp.Queue(), mp.Queue(), True, False, state_path, 2, "", False)
        assert new_trainer._model.model.state_dict() == dict_to_save["model"]


def test_send_model_state_to_server():
    response_queue = mp.Queue()
    query_queue = mp.Queue()
    trainer = get_mock_trainer(query_queue, response_queue, False, False, None, 1, "", False)
    trainer.send_model_state_to_server()
    response = response_queue.get()
    file_like = BytesIO(response)
    assert torch.load(file_like) == {
        "model": OrderedDict([("_weight", torch.tensor([1.0]))]),
        "optimizer-default": {
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


def test_send_status_to_server():
    response_queue = mp.Queue()
    query_queue = mp.Queue()
    trainer = get_mock_trainer(query_queue, response_queue, False, False, None, 1, "", False)
    trainer.send_status_to_server(20)
    response = response_queue.get()
    assert response["num_batches"] == 20
    assert response["num_samples"] == 0


@patch(
    "modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders",
    mock_get_dataloaders,
)
def test_train_invalid_query_message():
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    trainer = get_mock_trainer(query_status_queue, status_queue, False, False, None, 1, "", False)
    query_status_queue.put("INVALID MESSAGE")
    timeout = 5
    elapsed = 0
    while query_status_queue.empty():
        sleep(1)
        elapsed += 1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    with pytest.raises(ValueError, match="Unknown message in the status query queue"):
        trainer.train()

    elapsed = 0
    while not (query_status_queue.empty() and status_queue.empty()):
        sleep(1)
        elapsed += 1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")


# # pylint: disable=too-many-locals


@patch(
    "modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders",
    mock_get_dataloaders,
)
@patch.object(BaseCallback, "on_train_begin", return_value=None)
@patch.object(BaseCallback, "on_train_end", return_value=None)
@patch.object(BaseCallback, "on_batch_begin", return_value=None)
@patch.object(BaseCallback, "on_batch_end", return_value=None)
@patch.object(BaseCallback, "on_batch_before_update", return_value=None)
@patch.object(MetadataCollector, "send_metadata", return_value=None)
@patch.object(MetadataCollector, "cleanup", return_value=None)
@patch.object(CustomLRScheduler, "step", return_value=None)
def test_train(
    test_step,
    test_cleanup,
    test_send_metadata,
    test_on_batch_before_update,
    test_on_batch_end,
    test_on_batch_begin,
    test_on_train_end,
    test_on_train_begin,
):
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    trainer = get_mock_trainer(query_status_queue, status_queue, False, False, None, 2, "custom", False)
    query_status_queue.put(TrainerMessages.STATUS_QUERY_MESSAGE)
    timeout = 2
    elapsed = 0

    while query_status_queue.empty():
        sleep(0.1)
        elapsed += 0.1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    trainer.train()
    assert trainer._num_samples == 800
    elapsed = 0
    while not query_status_queue.empty():
        sleep(0.1)
        elapsed += 0.1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    assert test_on_train_begin.call_count == len(trainer._callbacks)
    assert test_on_train_end.call_count == len(trainer._callbacks)
    assert test_on_batch_begin.call_count == len(trainer._callbacks) * 100
    assert test_on_batch_end.call_count == len(trainer._callbacks) * 100
    assert test_on_batch_before_update.call_count == len(trainer._callbacks) * 100
    assert test_send_metadata.call_count == len(trainer._callbacks)
    test_cleanup.assert_called_once()

    if not platform.system() == "Darwin":
        assert status_queue.qsize() == 1
    else:
        assert not status_queue.empty()
    elapsed = 0
    while True:
        if not platform.system() == "Darwin":
            if status_queue.qsize() == 1:
                break
        else:
            if not status_queue.empty():
                break

        sleep(0.1)
        elapsed += 0.1

        if elapsed >= timeout:
            raise AssertionError("Did not reach desired queue state after 5 seconds.")

        status = status_queue.get()
        assert status["num_batches"] == 0
        assert status["num_samples"] == 0
        status_state = torch.load(io.BytesIO(status["state"]))
        checkpointed_state = {
            "model": OrderedDict(
                [
                    ("moda._weight", torch.tensor([1.0])),
                    ("modb._weight", torch.tensor([1.0])),
                    ("modc._weight", torch.tensor([1.0])),
                ]
            ),
            "optimizer-opt1": {
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
            "optimizer-opt2": {
                "state": {},
                "param_groups": [
                    {
                        "lr": 0.5,
                        "betas": (0.9, 0.999),
                        "eps": 1e-08,
                        "weight_decay": 0,
                        "amsgrad": False,
                        "maximize": False,
                        "foreach": None,
                        "capturable": False,
                        "differentiable": False,
                        "fused": False,
                        "params": [0],
                    },
                    {
                        "lr": 0.8,
                        "betas": (0.9, 0.999),
                        "eps": 1e-08,
                        "weight_decay": 0,
                        "amsgrad": False,
                        "maximize": False,
                        "foreach": None,
                        "capturable": False,
                        "differentiable": False,
                        "fused": False,
                        "params": [1],
                    },
                ],
            },
        }
        assert status_state == checkpointed_state
        assert os.path.exists(trainer._final_checkpoint_path / "model_final.modyn")
        final_state = torch.load(trainer._final_checkpoint_path / "model_final.modyn")
        assert final_state == checkpointed_state


@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch("modyn.trainer_server.internal.utils.training_info.dynamic_module_import")
@patch(
    "modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders",
    mock_get_dataloaders,
)
@patch.object(PytorchTrainer, "connect_to_selector", return_value=None)
@patch.object(PytorchTrainer, "get_selection_strategy", return_value=(False, "", {}))
def test_create_trainer_with_exception(
    test_selector_connection,
    test_election_strategy,
    test_dynamic_module_import,
    test_insecure_channel,
    test_grpc_connection_established,
):
    test_dynamic_module_import.return_value = MockModule(1)
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    exception_queue = mp.Queue()
    training_info = get_training_info(0, False, False, None, "", "", 1, "", False)
    query_status_queue.put("INVALID MESSAGE")
    timeout = 5
    elapsed = 0
    while query_status_queue.empty():
        sleep(1)
        elapsed += 1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    with tempfile.NamedTemporaryFile() as temp:
        train(training_info, "cpu", temp.name, exception_queue, query_status_queue, status_queue)
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

        assert pathlib.Path(temp.name).exists()
