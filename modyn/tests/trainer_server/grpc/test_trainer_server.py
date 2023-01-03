from unittest.mock import patch

import pytest
import json
import torch


from modyn.trainer_server.grpc.trainer_server import TrainerGRPCServer
from modyn.trainer_server.grpc.trainer_server_pb2 import Data, JsonString, RegisterTrainServerRequest, StartTrainingRequest

dummy_register_request = RegisterTrainServerRequest(
    training_id=1,
    model_id="test",
    batch_size=32,
    torch_optimizer='SGD',
    optimizer_parameters=JsonString(value=json.dumps({'lr': 0.1})),
    model_configuration=JsonString(value=json.dumps({'size': 10})),
    data_info=Data(
        dataset_id="Dataset",
        num_dataloaders=1
    ),
)

dummy_start_training_request = StartTrainingRequest(
    training_id=1,
    load_checkpoint_path="test"
)


class DummyDataset(torch.utils.data.dataset.Dataset):
    def __init__(self) -> None:
        super().__init__()


class DummyModel():
    def __init__(self) -> None:
        pass

    def train(self, path1: str, path2: str) -> None:
        pass


@patch('modyn.trainer_server.grpc.trainer_server.prepare_dataloaders')
def test_register_no_dataloader(test_prepare_dataloaders):

    trainer_server = TrainerGRPCServer()

    test_prepare_dataloaders.return_value = (None, None)
    response = trainer_server.register(dummy_register_request, None)

    assert response.success is False
    assert trainer_server._training_dict == {}


@patch('modyn.trainer_server.grpc.trainer_server.get_model')
@patch('modyn.trainer_server.grpc.trainer_server.prepare_dataloaders')
def test_register(test_prepare_dataloaders, test_get_model):

    trainer_server = TrainerGRPCServer()
    model = DummyModel()

    test_prepare_dataloaders.return_value = (torch.utils.data.DataLoader(DummyDataset()), None)
    test_get_model.return_value = model
    response = trainer_server.register(dummy_register_request, None)

    assert response.success is True
    assert trainer_server._training_dict == {1: model}


def test_start_training_not_registered():

    trainer_server = TrainerGRPCServer()
    with pytest.raises(ValueError):
        trainer_server.start_training(dummy_start_training_request, None)


def test_start_training():

    trainer_server = TrainerGRPCServer()
    model = DummyModel()

    trainer_server._training_dict[1] = model

    trainer_server.start_training(dummy_start_training_request, None)

    assert 1 in trainer_server._training_process_dict
