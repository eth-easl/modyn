from io import BytesIO
import os
import tempfile
import time
from unittest.mock import patch

import pytest
import json
import torch
import multiprocessing as mp
from unittest import mock

from modyn.trainer_server.grpc.trainer_server import TrainerGRPCServer
from modyn.trainer_server.grpc.generated.trainer_server_pb2 import CheckpointInfo, Data, JsonString, RegisterTrainServerRequest, StartTrainingRequest, TrainerAvailableRequest, TrainingStatusRequest
from modyn.trainer_server.utils.training_utils import TrainingProcessInfo, STATUS_QUERY_MESSAGE, TrainingInfo

start_training_request = StartTrainingRequest(
    training_id=1,
    trigger_point="new",
    load_checkpoint_path="test"
)
trainer_available_request = TrainerAvailableRequest()

register_request = RegisterTrainServerRequest(
    training_id=1,
    model_id="test",
    batch_size=32,
    torch_optimizer='SGD',
    torch_criterion='CrossEntropyLoss',
    optimizer_parameters=JsonString(value=json.dumps({'lr': 0.1})),
    model_configuration=JsonString(value=json.dumps({})),
    criterion_parameters=JsonString(value=json.dumps({})),
    data_info=Data(
        dataset_id="Dataset",
        num_dataloaders=1
    ),
    checkpoint_info=CheckpointInfo(
        checkpoint_interval=10,
        checkpoint_path="/tmp"
    ),
    transform_list=[]
)

get_status_request = TrainingStatusRequest(training_id=1)

def foo():
    return

def get_training_process_info():
    status_query_queue = mp.Queue()
    status_response_queue = mp.Queue()
    exception_queue = mp.Queue()

    training_process_info = TrainingProcessInfo(
        mp.Process(),
        exception_queue,
        status_query_queue,
        status_response_queue,
    )
    return training_process_info

@patch('modyn.trainer_server.utils.training_utils.hasattr', return_value=True)
def get_training_info(temp, test_hasattr):
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
        checkpoint_info=CheckpointInfo(checkpoint_interval=10, checkpoint_path=temp.name)
    )
    training_info = TrainingInfo(request)
    return training_info

def test_trainer_available():
    trainer_server = TrainerGRPCServer()
    response = trainer_server.trainer_available(trainer_available_request, None)
    assert response.available == True

@patch.object(mp.Process, 'is_alive', return_value=True)
def test_trainer_not_available(test_is_alive):
    trainer_server = TrainerGRPCServer()
    trainer_server._training_process_dict[10] = TrainingProcessInfo(mp.Process(), mp.Queue(), mp.Queue(), mp.Queue())
    response = trainer_server.trainer_available(trainer_available_request, None)
    assert response.available == False

@patch('modyn.trainer_server.utils.training_utils.hasattr', return_value=False)
def test_register_invalid(test_hasattr):
    trainer_server = TrainerGRPCServer()
    with pytest.raises(ValueError, match=r"Model \w+ not available"):
        trainer_server.register(register_request, None)

@patch('modyn.trainer_server.utils.training_utils.hasattr', return_value=True)
def test_register(test_hasattr):
    trainer_server = TrainerGRPCServer()
    response = trainer_server.register(register_request, None)
    assert response.success is True
    assert register_request.training_id in trainer_server._training_dict

def test_start_training_not_registered():

    trainer_server = TrainerGRPCServer()
    with pytest.raises(ValueError, match=r"Training with id \d has not been registered"):
        trainer_server.start_training(start_training_request, None)


def test_start_training():
    trainer_server = TrainerGRPCServer()
    m = mock.Mock()
    m.side_effect = foo
    trainer_server._training_dict[1] = None
    with patch("multiprocessing.Process.start", m):
        trainer_server.start_training(start_training_request, None)
        assert 1 in trainer_server._training_process_dict

def test_get_training_status_not_registered():
    trainer_server = TrainerGRPCServer()
    with pytest.raises(ValueError, match=r"Training with id \d has not been registered"):
        trainer_server.get_training_status(get_status_request, None)

@patch.object(mp.Process, 'is_alive', return_value=True)
@patch.object(TrainerGRPCServer, 'get_status', return_value=(b'state', 10))
@patch.object(TrainerGRPCServer, 'get_child_exception')
@patch.object(TrainerGRPCServer, 'get_latest_checkpoint')
def test_get_training_status_alive(test_get_latest_checkpoint, test_get_child_exception, test_get_status, test_is_alive):
    trainer_server = TrainerGRPCServer()
    training_process_info = get_training_process_info()
    trainer_server._training_process_dict[1] = training_process_info
    trainer_server._training_dict[1] = None

    response = trainer_server.get_training_status(get_status_request, None)
    assert response.is_running == True
    assert response.state_available == True
    assert response.iteration == 10
    assert response.state == b'state'
    test_get_latest_checkpoint.assert_not_called()
    test_get_child_exception.assert_not_called()

@patch.object(mp.Process, 'is_alive', return_value=False)
@patch.object(TrainerGRPCServer, 'get_latest_checkpoint', return_value=(b'state', 10))
@patch.object(TrainerGRPCServer, 'get_child_exception', return_value="exception")
@patch.object(TrainerGRPCServer, 'get_status')
def test_get_training_status_finished_with_exception(test_get_status, test_get_child_exception, test_get_latest_checkpoint, test_is_alive):
    trainer_server = TrainerGRPCServer()
    training_process_info = get_training_process_info()
    trainer_server._training_process_dict[1] = training_process_info
    trainer_server._training_dict[1] = None

    response = trainer_server.get_training_status(get_status_request, None)
    assert response.is_running == False
    assert response.state_available == True
    assert response.iteration == 10
    assert response.state == b'state'
    assert response.exception == "exception"
    test_get_status.assert_not_called()

@patch.object(mp.Process, 'is_alive', return_value=False)
@patch.object(TrainerGRPCServer, 'get_latest_checkpoint', return_value=(None, -1))
@patch.object(TrainerGRPCServer, 'get_child_exception', return_value="exception")
@patch.object(TrainerGRPCServer, 'get_status')
def test_get_training_status_finished_no_checkpoint(test_get_status, test_get_child_exception, test_get_latest_checkpoint, test_is_alive):
    trainer_server = TrainerGRPCServer()
    training_process_info = get_training_process_info()
    trainer_server._training_process_dict[1] = training_process_info
    trainer_server._training_dict[1] = None

    response = trainer_server.get_training_status(get_status_request, None)
    assert response.is_running == False
    assert response.state_available == False
    assert response.exception == "exception"
    test_get_status.assert_not_called()

def test_get_status():
    trainer_server = TrainerGRPCServer()
    state_dict = {
        'state': {},
        'iteration': 10
    }

    training_process_info = get_training_process_info()
    trainer_server._training_process_dict[1] = training_process_info
    training_process_info.status_response_queue.put(state_dict)
    state, iteration = trainer_server.get_status(1)
    assert state == state_dict['state']
    assert iteration == state_dict['iteration']
    assert training_process_info.status_query_queue.qsize() == 1
    assert training_process_info.status_response_queue.empty()
    query = training_process_info.status_query_queue.get()
    assert query == STATUS_QUERY_MESSAGE

def test_get_child_exception_not_found():
    trainer_server = TrainerGRPCServer()
    training_process_info = get_training_process_info()
    trainer_server._training_process_dict[1] = training_process_info
    child_exception = trainer_server.get_child_exception(1)
    assert child_exception is None

def test_get_child_exception():
    trainer_server = TrainerGRPCServer()
    training_process_info = get_training_process_info()
    trainer_server._training_process_dict[1] = training_process_info

    exception_msg="exception"
    training_process_info.exception_queue.put(exception_msg)

    child_exception = trainer_server.get_child_exception(1)
    assert child_exception==exception_msg

def test_get_latest_checkpoint_not_found():
    trainer_server = TrainerGRPCServer()
    trainer_server._training_dict[1] = get_training_info(tempfile.TemporaryDirectory())

    training_state, iteration = trainer_server.get_latest_checkpoint(1)
    assert training_state is None
    assert iteration==-1

def test_get_latest_checkpoint():
    trainer_server = TrainerGRPCServer()
    temp = tempfile.TemporaryDirectory()

    training_info = get_training_info(temp)
    trainer_server._training_dict[1] = training_info

    dict_to_save={
        'state': {'weight': 10},
        'iteration': 10
    }

    checkpoint_file = training_info.checkpoint_path + '/checkp'
    torch.save(dict_to_save, checkpoint_file)

    training_state, iteration = trainer_server.get_latest_checkpoint(1)
    assert iteration == 10

    dict_to_save.pop('iteration')
    assert torch.load(BytesIO(training_state))['state'] == dict_to_save['state']
