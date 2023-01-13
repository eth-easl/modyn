# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
from collections import OrderedDict
import io
import json
from unittest.mock import patch, MagicMock
import pytest
import torch
import os
import multiprocessing as mp
import tempfile
from io import BytesIO

from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    CheckpointInfo,
    Data,
    JsonString,
    RegisterTrainServerRequest
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


def mock_get_dataloaders(training_id, dataset_id, num_dataloaders, batch_size, transform_list, sample_id):
    mock_train_dataloader = iter(
        [(torch.ones(8, 10, requires_grad=True), torch.ones(8, dtype=int)) for _ in range(100)]
    )
    return mock_train_dataloader, None


@patch('modyn.trainer_server.internal.utils.training_info.dynamic_module_import')
def get_training_info(dynamic_module_patch: MagicMock):
    dynamic_module_patch.return_value = MockModule()
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


@patch('modyn.trainer_server.internal.utils.training_info.dynamic_module_import')
def get_mock_trainer(query_queue: mp.Queue(), response_queue: mp.Queue(), dynamic_module_patch: MagicMock):
    dynamic_module_patch.return_value = MockModule()
    training_info = get_training_info()
    trainer = PytorchTrainer(training_info, 'cpu', "new", query_queue, response_queue)
    return trainer


def test_trainer_init():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue())
    assert isinstance(trainer._model, MockModelWrapper)
    assert isinstance(trainer._optimizer, torch.optim.SGD)
    assert isinstance(trainer._criterion, torch.nn.CrossEntropyLoss)
    assert trainer._device == 'cpu'
    assert trainer._num_samples == 0
    assert trainer._checkpoint_interval == 10
    assert trainer._checkpoint_path == "checkpoint_test"
    assert os.path.isdir(trainer._checkpoint_path)


def test_save_state_to_file():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue())
    with tempfile.NamedTemporaryFile() as temp:
        trainer.save_state(temp.name, 10)
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


def test_save_state_to_buffer():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue())
    buffer = io.BytesIO()
    trainer.save_state(buffer)
    buffer.seek(0)
    saved_dict = torch.load(buffer)
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
    }


def test_load_checkpoint():
    trainer = get_mock_trainer(mp.Queue(), mp.Queue())

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


def test_send_state_to_server():
    response_queue = mp.Queue()
    query_queue = mp.Queue()
    trainer = get_mock_trainer(query_queue, response_queue)
    trainer.send_state_to_server(20)
    response = response_queue.get()
    assert response['num_batches'] == 20
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


@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders", mock_get_dataloaders)
def test_train_invalid_query_message():
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    trainer = get_mock_trainer(query_status_queue, status_queue)
    query_status_queue.put("INVALID MESSAGE")
    with tempfile.NamedTemporaryFile() as temp:
        with pytest.raises(ValueError, match="Unknown message in the status query queue"):
            trainer.train(temp.name)
        assert query_status_queue.qsize() == 0
        assert status_queue.qsize() == 0


@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders", mock_get_dataloaders)
def test_train():
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    trainer = get_mock_trainer(query_status_queue, status_queue)
    query_status_queue.put(TrainerMessages.STATUS_QUERY_MESSAGE)
    with tempfile.NamedTemporaryFile() as temp:
        trainer.train(temp.name)
        assert os.path.exists(temp.name)
        assert trainer._num_samples == 800
        assert query_status_queue.qsize() == 0
        assert status_queue.qsize() == 1
        status = status_queue.get()
        assert status['num_batches'] == 0
        assert status['num_samples'] == 0
        status_state = torch.load(io.BytesIO(status['state']))
        assert status_state == {
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
        }


@patch('modyn.trainer_server.internal.utils.training_info.dynamic_module_import')
@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.prepare_dataloaders", mock_get_dataloaders)
def test_create_trainer_with_exception(test_dynamic_module_import):
    test_dynamic_module_import.return_value = MockModule()
    query_status_queue = mp.Queue()
    status_queue = mp.Queue()
    exception_queue = mp.Queue()
    training_info = get_training_info()
    query_status_queue.put("INVALID MESSAGE")
    train(training_info, 'cpu', 'log_file', None, 'new', exception_queue, query_status_queue, status_queue)
    assert query_status_queue.empty()
    assert status_queue.empty()
    assert exception_queue.qsize() == 1
    exception = exception_queue.get()
    assert "ValueError: Unknown message in the status query queue" in exception
