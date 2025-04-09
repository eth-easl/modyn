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
from unittest.mock import MagicMock, call, patch

import grpc
import pytest
import torch
import transformers

from modyn.config import ModynConfig
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.trainer_server.internal.dataset.key_sources import SelectorKeySource
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
from modyn.trainer_server.internal.trainer.remote_downsamplers import RemoteGradMatchDownsamplingStrategy
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo
from modyn.utils import DownsamplingMode


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

    def forward(self, data, sample_ids=None):
        return data


class MockSuperModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moda = MockModel()
        self.modb = MockModel()
        self.modc = MockModel()

    def forward(self, data, sample_ids=None):
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


class MockDataloader:
    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.dataset = MagicMock()

    def __iter__(self):
        return iter(
            [
                (
                    ("1",) * self.batch_size,
                    torch.ones(self.batch_size, 10, requires_grad=True),
                    torch.ones(self.batch_size, dtype=torch.uint8),
                )
                for _ in range(self.num_batches)
            ]
        )

    def __len__(self):
        return self.num_batches


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
    prefetched_partitions,
    num_parallel_requests,
    shuffle,
    tokenizer,
    log_path,
    drop_last,
    include_labels=True,
    num_batches: int = 100,
    **kwargs,
):
    mock_train_dataloader = MockDataloader(batch_size, num_batches)
    return mock_train_dataloader, None


def noop_constructor_mock(self, channel):
    pass


# # pylint: disable=too-many-locals
@patch("modyn.trainer_server.internal.utils.training_info.dynamic_module_import")
def get_training_info(
    training_id: int,
    batch_size: int,
    use_pretrained: bool,
    load_optimizer_state: bool,
    pretrained_model_path: pathlib.Path,
    storage_address: str,
    selector_address: str,
    num_optimizers: int,
    lr_scheduler: str,
    transform_label: bool,
    offline_dataset_path: str,
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
                "algorithm": "AdamW",
                "source": "HuggingFace",
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
            "step_every": "batch",
            "optimizers": ["default"] if num_optimizers == 1 else ["opt1"],
            "config": {"step_size": 10},
        }
    elif lr_scheduler == "custom":
        lr_scheduler_config = {
            "name": "custom_scheduler",
            "source": "Custom",
            "step_every": "batch",
            "optimizers": ["default"] if num_optimizers == 1 else ["opt1", "opt2"],
            "config": {},
        }
    elif lr_scheduler == "torch_cosine":
        lr_scheduler_config = {
            "name": "CosineAnnealingLR",
            "source": "PyTorch",
            "step_every": "batch",
            "optimizers": ["default"] if num_optimizers == 1 else ["opt1"],
            "config": {"T_max": "MODYN_NUM_BATCHES"},
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
                data_info=Data(dataset_id="MNIST", num_dataloaders=2),
                torch_optimizers_configuration=JsonString(value=json.dumps(torch_optimizers_configuration)),
                criterion_parameters=JsonString(value=json.dumps({})),
                batch_size=batch_size,
                torch_criterion="CrossEntropyLoss",
                checkpoint_info=CheckpointInfo(checkpoint_interval=10, checkpoint_path=tmpdirname),
                bytes_parser=PythonString(value=get_mock_bytes_parser()),
                transform_list=[],
                use_pretrained_model=use_pretrained,
                load_optimizer_state=load_optimizer_state,
                pretrained_model_id=1 if use_pretrained else -1,
                lr_scheduler=JsonString(value=json.dumps(lr_scheduler_config)),
                label_transformer=PythonString(value=get_mock_label_transformer() if transform_label else ""),
                grad_scaler_configuration=JsonString(value=json.dumps({})),
                epochs_per_trigger=1,
            )
            training_info = TrainingInfo(
                request,
                training_id,
                "model",
                json.dumps({}),
                False,
                storage_address,
                selector_address,
                offline_dataset_path,
                pathlib.Path(final_tmpdirname),
                pathlib.Path(final_tmpdirname) / "log.log",
                pretrained_model_path,
            )
            return training_info


@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders", mock_get_dataloaders)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch("modyn.trainer_server.internal.utils.training_info.dynamic_module_import")
@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.dynamic_module_import")
@patch.object(PytorchTrainer, "connect_to_selector", return_value=None)
@patch.object(PytorchTrainer, "get_selection_strategy")
@patch.object(PytorchTrainer, "get_num_samples_in_trigger")
@patch.object(SelectorKeySource, "uses_weights", return_value=False)
def get_mock_trainer(
    modyn_config: ModynConfig,
    query_queue_training: mp.Queue,
    response_queue_training: mp.Queue,
    use_pretrained: bool,
    load_optimizer_state: bool,
    pretrained_model_path: pathlib.Path,
    num_optimizers: int,
    lr_scheduler: str,
    transform_label: bool,
    mock_weights: MagicMock,
    mock_get_num_samples: MagicMock,
    mock_selection_strategy: MagicMock,
    mock_selector_connection: MagicMock,
    lr_scheduler_dynamic_module_patch: MagicMock,
    model_dynamic_module_patch: MagicMock,
    test_insecure_channel: MagicMock,
    test_grpc_connection_established_selector: MagicMock,
    test_grpc_connection_established: MagicMock,
    batch_size: int = 32,
    selection_strategy: tuple[bool, str, dict] = (False, "", {}),
):
    model_dynamic_module_patch.return_value = MockModule(num_optimizers)
    lr_scheduler_dynamic_module_patch.return_value = MockLRSchedulerModule()
    mock_get_num_samples.return_value = batch_size * 100

    mock_selection_strategy.return_value = selection_strategy

    training_info = get_training_info(
        0,
        batch_size,
        use_pretrained,
        load_optimizer_state,
        pretrained_model_path,
        "",
        "",
        num_optimizers,
        lr_scheduler,
        transform_label,
        "/tmp/offline_dataset",
    )
    trainer = PytorchTrainer(
        modyn_config.model_dump(by_alias=True),
        training_info,
        "cpu",
        query_queue_training,
        response_queue_training,
        mp.Queue(),
        mp.Queue(),
        logging.getLogger(__name__),
    )
    return trainer


def test_trainer_init(dummy_system_config: ModynConfig):
    trainer = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), False, False, None, 1, "", False)
    assert isinstance(trainer._model, MockModelWrapper)
    assert len(trainer._optimizers) == 1
    assert isinstance(trainer._optimizers["default"], torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert not trainer._lr_scheduler
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    assert trainer._label_transformer_function is None


def test_trainer_init_multi_optimizers(dummy_system_config: ModynConfig):
    trainer = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), False, False, b"state", 2, "", False)
    assert isinstance(trainer._model, MockSuperModelWrapper)
    assert len(trainer._optimizers) == 2
    assert isinstance(trainer._optimizers["opt1"], torch.optim.SGD)
    assert isinstance(trainer._optimizers["opt2"], transformers.AdamW)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert not trainer._lr_scheduler
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    assert trainer._label_transformer_function is None


def test_trainer_init_torch_lr_scheduler(dummy_system_config: ModynConfig):
    trainer = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), False, False, None, 1, "torch", False)
    assert isinstance(trainer._model, MockModelWrapper)
    assert len(trainer._optimizers) == 1
    assert isinstance(trainer._optimizers["default"], torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(trainer._lr_scheduler, torch.optim.lr_scheduler.StepLR)
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    assert trainer._label_transformer_function is None


def test_trainer_init_custom_lr_scheduler(dummy_system_config: ModynConfig):
    trainer = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), False, False, None, 1, "custom", False)
    assert isinstance(trainer._model, MockModelWrapper)
    assert len(trainer._optimizers) == 1
    assert isinstance(trainer._optimizers["default"], torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(trainer._lr_scheduler, CustomLRScheduler)
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    assert trainer._label_transformer_function is None


@patch.object(PytorchTrainer, "load_state_if_given")
def test_trainer_init_from_pretrained_model(load_state_if_given_mock, dummy_system_config: ModynConfig):
    trainer = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), True, False, "/path/to/model", 1, "", False)
    assert isinstance(trainer._model, MockModelWrapper)
    assert isinstance(trainer._optimizers["default"], torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    load_state_if_given_mock.assert_called_once_with("/path/to/model", False)
    assert trainer._label_transformer_function is None


def test_trainer_init_with_label_transformer(dummy_system_config: ModynConfig):
    trainer = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), False, False, None, 1, "", True)
    assert isinstance(trainer._model, MockModelWrapper)
    assert len(trainer._optimizers) == 1
    assert isinstance(trainer._optimizers["default"], torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert not trainer._lr_scheduler
    assert trainer._device == "cpu"
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert os.path.isdir(trainer._checkpoint_path)
    assert trainer._label_transformer_function is not None
    test_tensor = torch.ones(10, dtype=torch.int32)
    assert torch.equal(trainer._label_transformer_function(test_tensor), torch.ones(10, dtype=torch.float32))
    assert trainer._label_transformer_function(test_tensor).dtype == torch.float32


def test_gradient_accumulation_equivalence(dummy_system_config):
    batch_size = 32

    # Trainer A: gradient accumulation (process 2 mini-batches of size `batch_size`)
    trainer_accum = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), False, False, None, 2, "", False)

    trainer_accum.gradient_accumulation_steps = 2
    # Process exactly 2 mini-batches (2*batch_size samples) before stopping training.
    trainer_accum.num_samples_to_pass = batch_size * 2
    trainer_accum._train_dataloader = MockDataloader(batch_size=batch_size, num_batches=2)

    # Trainer B: no gradient accumulation, processing one batch of size 2*batch_size
    trainer_single = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), False, False, None, 2, "", False)
    trainer_single.gradient_accumulation_steps = 1
    trainer_single.num_samples_to_pass = batch_size * 2
    trainer_single._train_dataloader = MockDataloader(batch_size=batch_size * 2, num_batches=1)

    def clone_params(model):
        return {name: param.detach().clone() for name, param in model.named_parameters()}

    # Ensure both trainers start with the same initial parameters.
    init_params = clone_params(trainer_accum._model.model)

    trainer_accum.train()
    trainer_single.train()

    # Instead of checking the logged "num_batches_trained" (which counts mini-batches),
    # we compare the final model parameters to ensure they are equivalent.
    final_params_accum = clone_params(trainer_accum._model.model)
    final_params_single = clone_params(trainer_single._model.model)

    for name in init_params.keys():
        assert torch.allclose(final_params_accum[name], final_params_single[name], atol=1e-5), f"Mismatch in {name}"


def test_save_state_to_file(dummy_system_config: ModynConfig):
    trainer = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), False, False, None, 2, "", False)
    with tempfile.NamedTemporaryFile() as temp:
        trainer.save_state(pathlib.Path(temp.name), 10)
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
                    "lr": 0.5,
                    "betas": (0.9, 0.999),
                    "eps": 1e-06,
                    "weight_decay": 0.0,
                    "correct_bias": True,
                    "params": [0],
                },
                {
                    "lr": 0.8,
                    "betas": (0.9, 0.999),
                    "eps": 1e-06,
                    "weight_decay": 0.0,
                    "correct_bias": True,
                    "params": [1],
                },
            ],
        },
        "iteration": 10,
    }


def test_save_state_to_buffer(dummy_system_config: ModynConfig):
    trainer = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), False, False, None, 1, "", False)
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


def test_load_state_if_given(dummy_system_config: ModynConfig):
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

        trainer = get_mock_trainer(dummy_system_config, mp.Queue(), mp.Queue(), True, True, state_path, 2, "", False)
        assert trainer._model.model.state_dict() == dict_to_save["model"]
        assert trainer._optimizers["opt1"].state_dict() == dict_to_save["optimizer-opt1"]
        assert trainer._optimizers["opt2"].state_dict() == dict_to_save["optimizer-opt2"]
        initial_state_buffer.seek(0)
        state_path = pathlib.Path(tempdir) / "test.state"
        with open(state_path, "wb") as file:
            file.write(initial_state_buffer.read())

        new_trainer = get_mock_trainer(
            dummy_system_config, mp.Queue(), mp.Queue(), True, False, state_path, 2, "", False
        )
        assert new_trainer._model.model.state_dict() == dict_to_save["model"]


def test_send_model_state_to_server(dummy_system_config: ModynConfig):
    response_queue = mp.Queue()
    query_queue = mp.Queue()
    trainer = get_mock_trainer(dummy_system_config, query_queue, response_queue, False, False, None, 1, "", False)
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


@patch.object(PytorchTrainer, "weights_handling", return_value=(False, False))
def test_train_invalid_query_message(test_weight_handling, dummy_system_config: ModynConfig):
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    trainer = get_mock_trainer(dummy_system_config, query_status_queue, status_queue, False, False, None, 1, "", False)
    query_status_queue.put("INVALID MESSAGE")
    timeout = 5
    elapsed = 0
    while query_status_queue.empty():
        sleep(0.1)
        elapsed += 0.1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    with pytest.raises(ValueError, match="Unknown message in the status query queue"):
        trainer.train()

    elapsed = 0
    while not (query_status_queue.empty() and status_queue.empty()):
        sleep(0.1)
        elapsed += 0.1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")


# # pylint: disable=too-many-locals


@patch.object(BaseCallback, "on_train_begin", return_value=None)
@patch.object(BaseCallback, "on_train_end", return_value=None)
@patch.object(BaseCallback, "on_batch_begin", return_value=None)
@patch.object(BaseCallback, "on_batch_end", return_value=None)
@patch.object(BaseCallback, "on_batch_before_update", return_value=None)
@patch.object(MetadataCollector, "send_metadata", return_value=None)
@patch.object(MetadataCollector, "cleanup", return_value=None)
@patch.object(CustomLRScheduler, "step", return_value=None)
@patch.object(PytorchTrainer, "end_of_trigger_cleaning", return_value=None)
@patch.object(PytorchTrainer, "weights_handling", return_value=(False, False))
def test_train(
    test_weights_handling,
    test_cleaning,
    test_step,
    test_cleanup,
    test_send_metadata,
    test_on_batch_before_update,
    test_on_batch_end,
    test_on_batch_begin,
    test_on_train_end,
    test_on_train_begin,
    dummy_system_config: ModynConfig,
):
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    trainer = get_mock_trainer(
        dummy_system_config, query_status_queue, status_queue, False, False, None, 2, "custom", False, batch_size=8
    )
    query_status_queue.put(TrainerMessages.STATUS_QUERY_MESSAGE)
    query_status_queue.put(TrainerMessages.MODEL_STATE_QUERY_MESSAGE)
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

    elapsed = 0
    while True:
        if not platform.system() == "Darwin":
            if status_queue.qsize() == 2:
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
    # we didn't enable recording the training loss
    assert len(trainer._log["training_loss"]) == 0
    status_state = torch.load(io.BytesIO(status_queue.get()))
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
                    "betas": (0.9, 0.999),
                    "correct_bias": True,
                    "eps": 1e-06,
                    "lr": pytest.approx(0.5),
                    "params": [0],
                    "weight_decay": 0.0,
                },
                {
                    "betas": (0.9, 0.999),
                    "correct_bias": True,
                    "eps": 1e-06,
                    "lr": 0.8,
                    "params": [1],
                    "weight_decay": 0.0,
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
@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders", mock_get_dataloaders)
@patch.object(PytorchTrainer, "connect_to_selector", return_value=None)
@patch.object(PytorchTrainer, "get_selection_strategy", return_value=(False, "", {}))
@patch.object(PytorchTrainer, "weights_handling", return_value=(False, False))
@patch.object(PytorchTrainer, "get_num_samples_in_trigger", return_value=42)
def test_create_trainer_with_exception(
    test_get_num_samples_in_trigger,
    test_weighs_handling,
    test_selector_connection,
    test_election_strategy,
    test_dynamic_module_import,
    test_insecure_channel,
    test_grpc_connection_established,
    dummy_system_config: ModynConfig,
):
    test_dynamic_module_import.return_value = MockModule(1)
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    exception_queue = mp.Queue()
    training_info = get_training_info(0, 32, False, False, None, "", "", 1, "", False, "/tmp/offline_dataset")
    query_status_queue.put("INVALID MESSAGE")
    timeout = 5
    elapsed = 0
    while query_status_queue.empty():
        sleep(0.1)
        elapsed += 0.1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    with tempfile.NamedTemporaryFile() as temp:
        train(
            dummy_system_config.model_dump(by_alias=True),
            training_info,
            "cpu",
            pathlib.Path(temp.name),
            exception_queue,
            query_status_queue,
            status_queue,
            mp.Queue(),
            mp.Queue(),
        )
        elapsed = 0
        while not (query_status_queue.empty() and status_queue.empty()):
            sleep(0.1)
            elapsed += 0.1

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

            sleep(0.1)
            elapsed += 0.1

            if elapsed >= timeout:
                raise AssertionError("Did not reach desired queue state after 5 seconds.")

        exception = exception_queue.get()
        assert "ValueError: Unknown message in the status query queue" in exception

        assert pathlib.Path(temp.name).exists()


@pytest.mark.parametrize("downsampling_ratio, ratio_max", [(25, 100), (50, 100), (250, 1000), (125, 1000)])
@patch.object(BaseCallback, "on_train_begin", return_value=None)
@patch.object(BaseCallback, "on_train_end", return_value=None)
@patch.object(BaseCallback, "on_batch_begin", return_value=None)
@patch.object(BaseCallback, "on_batch_end", return_value=None)
@patch.object(BaseCallback, "on_batch_before_update", return_value=None)
@patch.object(MetadataCollector, "send_metadata", return_value=None)
@patch.object(MetadataCollector, "cleanup", return_value=None)
@patch.object(CustomLRScheduler, "step", return_value=None)
@patch.object(PytorchTrainer, "end_of_trigger_cleaning", return_value=None)
@patch.object(PytorchTrainer, "weights_handling", return_value=(False, True))
@patch.object(PytorchTrainer, "downsample_batch")
def test_train_batch_then_sample_accumulation(
    test_downsample_batch,
    test_weights_handling,
    test_cleaning,
    test_step,
    test_cleanup,
    test_send_metadata,
    test_on_batch_before_update,
    test_on_batch_end,
    test_on_batch_begin,
    test_on_train_end,
    test_on_train_begin,
    dummy_system_config: ModynConfig,
    downsampling_ratio,
    ratio_max,
):
    num_batches = 100  # hardcoded into mock dataloader
    batch_size = 32

    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    trainer = get_mock_trainer(
        dummy_system_config,
        query_status_queue,
        status_queue,
        False,
        True,
        None,
        2,
        "custom",
        False,
        batch_size=batch_size,
        selection_strategy=(
            True,
            "RemoteGradNormDownsampling",
            {"downsampling_ratio": downsampling_ratio, "sample_then_batch": False, "ratio_max": ratio_max},
        ),
    )
    assert trainer._downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE

    # Mock the downsample_batch method to return batches of the expected size
    expected_bts_size = int(batch_size * (downsampling_ratio / ratio_max))
    bts_accumulate_period = batch_size // expected_bts_size

    def mock_downsample_batch(data, sample_ids, target):
        mock_downsample_batch.num_downsamples += 1
        return (
            ((torch.ones(expected_bts_size, requires_grad=True) + mock_downsample_batch.num_downsamples) * len(data)),
            sample_ids[:expected_bts_size],
            target[:expected_bts_size],
            torch.ones(expected_bts_size),
        )

    mock_downsample_batch.num_downsamples = -1
    test_downsample_batch.side_effect = mock_downsample_batch

    # Mock the model's forward method to check the input data
    forward_calls = []

    def mock_forward(data, sample_ids=None):
        forward_calls.append(data)
        return torch.randn(data.shape[0], 10, requires_grad=True)  # Dummy output

    trainer._model.model.forward = mock_forward

    trainer.train()

    assert trainer._num_samples == batch_size * num_batches
    assert trainer._log["num_samples"] == batch_size * num_batches
    assert trainer._log["num_batches"] == num_batches
    # We only train on whole batches, hence we have to scale by batch size
    assert trainer._log["num_samples_trained"] == ((expected_bts_size * num_batches) // batch_size) * batch_size
    assert test_on_batch_begin.call_count == len(trainer._callbacks) * num_batches
    assert test_on_batch_end.call_count == len(trainer._callbacks) * num_batches
    assert test_downsample_batch.call_count == num_batches

    # Check if the model's forward method is called with the correctly accumulated data
    assert len(forward_calls) == num_batches // bts_accumulate_period
    for num_call, data in enumerate(forward_calls):
        assert data.shape[0] == batch_size

        range_end = (num_call + 1) * bts_accumulate_period + 1  # + 1 since end is exclusive
        range_start = range_end - bts_accumulate_period
        # We stack up zeros and add as much and 1 more as we add to the ones in the mocked downsampler
        expected_data = torch.cat([torch.zeros(expected_bts_size) + i for i in range(range_start, range_end, 1)])
        expected_data = expected_data * batch_size

        assert expected_data.shape[0] == data.shape[0]
        assert torch.allclose(data, expected_data)


@patch.object(MetadataCollector, "send_metadata", return_value=None)
@patch.object(MetadataCollector, "cleanup", return_value=None)
@patch.object(CustomLRScheduler, "step", return_value=None)
@patch.object(PytorchTrainer, "end_of_trigger_cleaning", return_value=None)
@patch.object(PytorchTrainer, "weights_handling", return_value=(False, False))
def test_lr_scheduler_init(
    test_weights_handling,
    test_cleaning,
    test_step,
    test_cleanup,
    test_send_metadata,
    dummy_system_config: ModynConfig,
):
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    # torch_cosine initializes a CosineAnnealingLR scheduler where T_max should be equal to the number of batches
    # Due to mock_get_num_samples.return_value = batch_size * 100 in get_mock_trainer, this is 100.

    trainer = get_mock_trainer(
        dummy_system_config,
        query_status_queue,
        status_queue,
        False,
        False,
        None,
        2,
        "torch_cosine",
        False,
        batch_size=8,
    )

    assert trainer._lr_scheduler.T_max == 100


@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.SelectorKeySource")
@patch.object(PytorchTrainer, "get_available_labels_from_selector")
@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_per_class_dataloader_from_online_dataset")
@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.LocalDatasetWriter")
@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.LocalKeySource")
@patch.object(PytorchTrainer, "start_embedding_recording_if_needed")
@patch.object(PytorchTrainer, "end_embedding_recorder_if_needed")
@patch.object(PytorchTrainer, "get_embeddings_if_recorded")
@patch.object(RemoteGradMatchDownsamplingStrategy, "inform_samples")
@patch.object(RemoteGradMatchDownsamplingStrategy, "inform_end_of_current_label")
@patch.object(PytorchTrainer, "update_queue")
def test_downsample_trigger_training_set_label_by_label(
    test_update_queue,
    test_inform_end_of_current_label,
    test_inform_samples,
    test_get_embeddings,
    test_end_embedding_recording,
    test_start_embedding_recording,
    test_local_key_source,
    test_local_dataset_writer,
    test_prepare_per_class_dataloader,
    test_get_available_labels,
    test_selector_key_source,
    dummy_system_config: ModynConfig,
):
    batch_size = 4
    available_labels = [0, 1, 2, 3, 4, 5]
    test_prepare_per_class_dataloader.return_value = MockDataloader(batch_size, 100)
    test_get_available_labels.return_value = available_labels
    num_batches = 100  # hardcoded into mock dataloader
    query_status_queue_training = mp.Queue()
    status_queue_training = mp.Queue()
    trainer = get_mock_trainer(
        dummy_system_config,
        query_status_queue_training,
        status_queue_training,
        False,
        False,
        None,
        2,
        "custom",
        False,
        batch_size=batch_size,
        selection_strategy=(
            True,
            "RemoteGradMatchDownsamplingStrategy",
            {
                "downsampling_ratio": 25,
                "downsampling_period": 1,
                "sample_then_batch": True,
                "balance": True,
                "full_grad_approximation": "LastLayer",
                "ratio_max": 100,
            },
        ),
    )
    assert trainer._downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH
    assert trainer._downsampler.requires_data_label_by_label
    trainer.downsample_trigger_training_set()
    assert test_prepare_per_class_dataloader.call_count == 1
    assert test_update_queue.call_count == len(available_labels) * num_batches + 1
    # check the args of the last call
    last_call_args = test_update_queue.call_args_list[-1]
    expected_batch_number = len(available_labels) * num_batches
    expected_num_samples = expected_batch_number * batch_size
    assert last_call_args == call("DOWNSAMPLING", expected_batch_number, expected_num_samples, training_active=True)
    assert test_inform_end_of_current_label.call_count == len(available_labels)


@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.SelectorKeySource")
@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.LocalDatasetWriter")
@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.LocalKeySource")
@patch.object(PytorchTrainer, "start_embedding_recording_if_needed")
@patch.object(PytorchTrainer, "end_embedding_recorder_if_needed")
@patch.object(PytorchTrainer, "get_embeddings_if_recorded")
@patch.object(RemoteGradMatchDownsamplingStrategy, "inform_samples")
@patch.object(RemoteGradMatchDownsamplingStrategy, "select_points", return_value=([1, 2], torch.ones(2)))
@patch.object(PytorchTrainer, "update_queue")
def test_downsample_trigger_training_set(
    test_update_queue,
    test_select_points,
    test_inform_samples,
    test_get_embeddings,
    test_end_embedding_recording,
    test_start_embedding_recording,
    test_local_key_source,
    test_local_dataset_writer,
    test_selector_key_source,
    dummy_system_config: ModynConfig,
):
    batch_size = 4
    num_batches = 100  # hardcoded into mock dataloader
    query_status_queue_training = mp.Queue()
    status_queue_training = mp.Queue()
    trainer = get_mock_trainer(
        dummy_system_config,
        query_status_queue_training,
        status_queue_training,
        False,
        False,
        None,
        2,
        "custom",
        False,
        batch_size=batch_size,
        selection_strategy=(
            True,
            "RemoteGradMatchDownsamplingStrategy",
            {
                "downsampling_ratio": 25,
                "downsampling_period": 1,
                "sample_then_batch": True,
                "balance": False,
                "full_grad_approximation": "LastLayer",
                "ratio_max": 100,
            },
        ),
    )
    assert trainer._downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH
    assert not trainer._downsampler.requires_data_label_by_label
    trainer.downsample_trigger_training_set()
    assert test_update_queue.call_count == num_batches + 1
    # check the args of the last call
    last_call_args = test_update_queue.call_args_list[-1]
    expected_batch_number = num_batches
    expected_num_samples = expected_batch_number * batch_size
    assert last_call_args == call("DOWNSAMPLING", expected_batch_number, expected_num_samples, training_active=True)
