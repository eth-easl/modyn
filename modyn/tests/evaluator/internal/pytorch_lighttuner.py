# pylint: disable=no-name-in-module
from __future__ import annotations


import io
import json
import logging
import os
import pathlib
import shutil
import tempfile
import traceback
from typing import Any
from unittest.mock import MagicMock, call, patch
from types import SimpleNamespace
import grpc
import pytest
import torch
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from torch.utils.data import IterableDataset
import copy
from modyn.evaluator.internal.utils.tuning_info import TuningInfo
from modyn.evaluator.internal.pytorch_lighttuner import PytorchTuner

from modyn.config import ModynConfig
from torch.utils.data import DataLoader, Dataset
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.trainer_server.internal.dataset.key_sources import SelectorKeySource
from modyn.trainer_server.custom_lr_schedulers.WarmupDecayLR.warmupdecay import WarmupDecayLR
from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector
from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.base_callback import BaseCallback
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    JsonString,
    DatasetInfo,
    PythonString
   
)

class MockStorageStub:
    def __init__(self, channel) -> None:
        pass

    def Get(self, request: GetRequest):  # pylint: disable=invalid-name
        for key in request.keys:
            yield GetResponse(
                samples=[key.to_bytes(2, "big"), (key + 1).to_bytes(2, "big")], keys=[key, key + 1], labels=[5, 5]
            )

    def GetDataPerWorker(self, request: GetDataPerWorkerRequest):  # pylint: disable=invalid-name
        for i in range(0, 8, 4):
            key = 8 * request.worker_id + i
            yield GetDataPerWorkerResponse(keys=[key, key + 2])


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
        "torch.Tensor:\n\treturn x"
    )


class MockModel(torch.nn.Module):
    def __init__(self, num_classes=10, input_dim=10):  
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)  # Adjusted to match data input shape

    def forward(self, data, sample_ids=None):
        data = data.to(torch.float32)  # Ensure float dtype for model input
        return self.fc(data)
    
class MockDataloader:
    def __init__(self, tuning_info, num_batches=20):
        self.batch_size = tuning_info.batch_size
        self.num_batches = num_batches
        self.dataset = MagicMock()

    def __iter__(self):
        return iter(
            [
                (
                    ("1",) * self.batch_size,  # sample_ids as strings (not used for training)
                    torch.ones(self.batch_size, 10, requires_grad=True),  # Input data as (batch_size, 10)
                    torch.randint(0, 10, (self.batch_size,), dtype=torch.long),  # Target as `long` for CE
                )
                for _ in range(self.num_batches)
            ]
        )

    def __len__(self):
        return self.num_batches
    
def mock_get_dataloader(self,tuning_info):
    """Creates a DataLoader similar to _prepare_dataloader."""
    mock_dataloader=MockDataloader(tuning_info)
    return mock_dataloader


def noop_constructor_mock(self, channel):
    pass



def get_tuning_info(
    evaluation_id: int,
    batch_size: int,
    num_optimizers: int,
    lr_scheduler: str,
    
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
            "step_every": "batch",
            "optimizers": ["default"] if num_optimizers == 1 else ["opt1"],
            "config": {"step_size": 10},
        }
    elif lr_scheduler == "custom":
        lr_scheduler_config = {
            "name": "WarmupDecayLR",
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
            "config": {"T_max": 100},
        }
    else:
        lr_scheduler_config = {}

    with tempfile.TemporaryDirectory() as tmpdirname:
        with tempfile.TemporaryDirectory() as final_tmpdirname:
            
            tuning_info = SimpleNamespace(
                    pipeline_id=1,
                    data_info=DatasetInfo(dataset_id="MNIST", num_dataloaders=2),
                    evaluation_id=42, 
                    batch_size=32,  
                    num_samples_to_pass=500,
                    transform_list=[],
                    bytes_parser=PythonString(value=get_mock_bytes_parser()),
                    torch_optimizers_configuration=JsonString(value=json.dumps(torch_optimizers_configuration)),
                    criterion_parameters=JsonString(value=json.dumps({})),
                    torch_criterion="CrossEntropyLoss",
                    lr_scheduler=JsonString(value=json.dumps(lr_scheduler_config)),
                    grad_scaler_configuration=JsonString(value=json.dumps({})),
                    epochs=10,  
                    label_transformer=PythonString(value=get_mock_label_transformer()),
                    device="cpu",
                    amp=False,
                    shuffle=True,  
                    enable_accurate_gpu_measurements=False,

                   
                    generative=False,
                    steps=100,  
                    drop_last_batch=False, 
                    
                    record_loss_every=10, 
                    seed=42,  
                    tokenizer=None, 
                    
                )

          
            return TuningInfo(
                tuning_info,
                1,
                None,
                None,

                
            )



def get_mock_tuner(
    modyn_config: ModynConfig,
    num_optimizers: int,
    lr_scheduler: str,
    
    lr_scheduler_dynamic_module_patch: MagicMock,
    model_dynamic_module_patch: MagicMock,
    batch_size: int,
    model:Any 

):
    model_dynamic_module_patch.return_value = MockModule(num_optimizers)
    lr_scheduler_dynamic_module_patch.return_value = MockLRSchedulerModule()
  
    tuning_info = get_tuning_info(
        0,
        batch_size,
        num_optimizers,
        lr_scheduler,
       
    )
    
    # Fixing argument order:
    tuner = PytorchTuner(
        tuning_info,  # ✅ Matches `tuning_info`
        logging.getLogger(__name__),  # ✅ Correctly placed `logger`
        model,  # ✅ Correctly assigned `model`
        "localhost:1234",
    )
    
    return tuner


def test_tuner_init(dummy_system_config: ModynConfig):
    tuner = get_mock_tuner(dummy_system_config,1,"",MagicMock(),MagicMock(),32,MockModel())

    # Ensure model initialization is correct
    assert isinstance(tuner._model, MockModel), "Expected tuner._model to be an instance of MockModule"

    # Validate optimizer setup
    assert len(tuner._optimizers) == 1, "Expected one optimizer to be initialized"
    assert isinstance(tuner._optimizers["default"], torch.optim.SGD), "Optimizer should be SGD"

    # Validate loss function
    assert isinstance(tuner._criterion, torch.nn.CrossEntropyLoss), "Loss function should be CrossEntropyLoss"

    # Ensure learning rate scheduler is disabled
    assert not tuner._lr_scheduler, "Expected no learning rate scheduler"

    # Verify device and batch size configurations
    assert tuner._device == "cpu", "Expected tuner to run on CPU"
    assert tuner._batch_size > 0, "Batch size should be greater than 0"

   
   



def test_tuner_init_multi_optimizers(dummy_system_config: ModynConfig):
    tuner = get_mock_tuner(dummy_system_config,2,"",MagicMock(),MagicMock(),32,MockSuperModel())
    assert isinstance(tuner._model, MockSuperModel)
    assert len(tuner._optimizers) == 2
    assert isinstance(tuner._optimizers["opt1"], torch.optim.SGD)
    assert isinstance(tuner._optimizers["opt2"], torch.optim.Adam)
    assert isinstance(tuner._criterion, torch.nn.CrossEntropyLoss)
    assert not tuner._lr_scheduler
    assert tuner._device == "cpu"
    assert tuner._batch_size > 0
    assert tuner._dataset_log_path is not None
   


def test_tuner_init_torch_lr_scheduler(dummy_system_config: ModynConfig):
    tuner = get_mock_tuner(dummy_system_config,1,"torch",MagicMock(),MagicMock(),32,MockModel())
    assert isinstance(tuner._model, MockModel)
    assert len(tuner._optimizers) == 1
    assert isinstance(tuner._optimizers["default"], torch.optim.SGD)
    assert isinstance(tuner._criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(tuner._lr_scheduler, torch.optim.lr_scheduler.StepLR)
    assert tuner._device == "cpu"
    assert tuner._batch_size > 0
   
   


def test_tuner_init_custom_lr_scheduler(dummy_system_config: ModynConfig):
    tuner = get_mock_tuner(dummy_system_config,1,"custom",MagicMock(),MagicMock(),32,MockModel())
    assert isinstance(tuner._model, MockModel)
    assert len(tuner._optimizers) == 1
    assert isinstance(tuner._optimizers["default"], torch.optim.SGD)
    assert isinstance(tuner._criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(tuner._lr_scheduler, WarmupDecayLR)
    assert tuner._device == "cpu"
    assert tuner._batch_size > 0






   
@patch("modyn.evaluator.internal.pytorch_lighttuner.PytorchTuner._prepare_dataloader", mock_get_dataloader)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.StorageStub", MockStorageStub)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.grpc_connection_established", return_value=True)
def test_tuner_light_tuning(
    
   
    
    dummy_system_config: ModynConfig,
):
    model=MockModel()
    tuner = get_mock_tuner(dummy_system_config,1,"",MagicMock(),MagicMock(),32,model)
    

    # Limit the tuning to 2 steps
    tuner._light_tuning_steps = 2

    # Capture old state before training
    old_model_state = copy.deepcopy(model.state_dict())
    old_optim_states = {
        name: copy.deepcopy(opt.state_dict()) for name, opt in tuner._optimizers.items()
    }

    # Run the light tuning process
    tuner.train()

    

    # Check that at least one model parameter changed
    new_model_state =model.state_dict()
    
    change_detected = any(
        not torch.equal(old_model_state[p], new_model_state[p]) for p in old_model_state
    )
    assert change_detected, "Expected at least one model parameter to change after light tuning."

    # Check that optimizer states changed
    new_optim_states = {
        name: opt.state_dict() for name, opt in tuner._optimizers.items()
    }
    for opt_name in old_optim_states:
        assert old_optim_states[opt_name] != new_optim_states[opt_name], \
            f"Expected optimizer {opt_name} state to change after light tuning."

