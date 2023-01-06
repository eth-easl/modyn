# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
from collections import OrderedDict
import json
from unittest.mock import patch, MagicMock
import torch
import os
import multiprocessing as mp
import tempfile
from io import BytesIO

from modyn.trainer_server.grpc.generated.trainer_server_pb2 import (
    CheckpointInfo,
    Data,
    JsonString,
    RegisterTrainServerRequest
)

from modyn.trainer_server.trainer.pytorch_trainer import PytorchTrainer
from modyn.trainer_server.utils.training_utils import TrainingInfo

query_queue = mp.Queue()
response_queue = mp.Queue()


class DummyModule:
    def __init__(self) -> None:
        self.model = DummyModelWrapper()


class DummyModelWrapper:
    def __init__(self, model_configuration=None) -> None:
        self.model = DummyModel()


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, data):
        return data


@patch('modyn.trainer_server.utils.training_utils.dynamic_module_import')
def get_training_info(dynamic_module_patch: MagicMock):
    dynamic_module_patch.return_value = DummyModule()
    request = RegisterTrainServerRequest(
        training_id=1,
        data_info=Data(dataset_id="MNIST", num_dataloaders=2),
        optimizer_parameters=JsonString(value=json.dumps({'lr': 0.1})),
        model_configuration=JsonString(value=json.dumps({})),
        criterion_parameters=JsonString(value=json.dumps({})),
        transform_list=[],
        model_id="model",
        torch_optimizer="SGD",
        batch_size=32,
        torch_criterion="CrossEntropyLoss",
        checkpoint_info=CheckpointInfo(checkpoint_interval=10, checkpoint_path="checkpoint_test")
    )
    training_info = TrainingInfo(request)
    return training_info


@patch('modyn.trainer_server.trainer.pytorch_trainer.get_model')
def get_dummy_trainer(test_get_model: MagicMock):
    test_get_model.return_value = DummyModelWrapper()
    training_info = get_training_info()
    trainer = PytorchTrainer(training_info, 'cpu', "new", query_queue, response_queue)
    return trainer


def test_save_checkpoint():
    trainer = get_dummy_trainer()
    with tempfile.NamedTemporaryFile() as temp:
        trainer.save_checkpoint(temp.name, 10)
        assert os.path.exists(temp.name)
        saved_dict = torch.load(temp.name)

    assert saved_dict == {
        'model': OrderedDict([('_weight', torch.tensor([1.]))]),
        'optimizer': {
            'state': {},
            'param_groups': [
                {
                    'lr': 0.1,
                    'momentum': 0,
                    'dampening': 0,
                    'weight_decay': 0,
                    'nesterov': False,
                    'maximize': False,
                    'foreach': None,
                    'differentiable': False,
                    'params': [0]
                }
            ]
        },
        'iteration': 10
    }


def test_load_checkpoint():
    trainer = get_dummy_trainer()

    dict_to_save = {
        'model': OrderedDict([('_weight', torch.tensor([100.]))]),
        'optimizer': {
            'state': {},
            'param_groups': [
                {
                    'lr': 0.1,
                    'momentum': 0,
                    'dampening': 0,
                    'weight_decay': 0,
                    'nesterov': False,
                    'maximize': False,
                    'foreach': None,
                    'differentiable': False,
                    'params': [0]
                }
            ]
        },
        'iteration': 100
    }

    with tempfile.NamedTemporaryFile() as temp:
        torch.save(dict_to_save, temp.name)
        trainer.load_checkpoint(temp.name)
        assert trainer._model.model.state_dict() == dict_to_save['model']
        assert trainer._optimizer.state_dict() == dict_to_save['optimizer']


def test_create_logger():
    trainer = get_dummy_trainer()
    with tempfile.NamedTemporaryFile() as temp:
        trainer.create_logger(temp.name)
        assert os.path.exists(temp.name)


def test_send_state_to_server():
    trainer = get_dummy_trainer()
    trainer.send_state_to_server(20)
    response = response_queue.get()
    assert response['iteration'] == 20
    file_like = BytesIO(response['state'])
    assert torch.load(file_like) == {
        'model': OrderedDict([('_weight', torch.tensor([1.]))]),
        'optimizer': {
            'state': {},
            'param_groups': [
                {
                    'lr': 0.1,
                    'momentum': 0,
                    'dampening': 0,
                    'weight_decay': 0,
                    'nesterov': False,
                    'maximize': False,
                    'foreach': None,
                    'differentiable': False,
                    'params': [0]
                }
            ]
        }
    }
