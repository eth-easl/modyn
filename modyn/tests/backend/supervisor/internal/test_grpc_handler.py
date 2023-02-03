# pylint: disable=unused-argument,no-value-for-parameter,no-name-in-module
from unittest.mock import patch

import grpc
import pytest
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import (
    DataInformRequest,
    JsonString,
    PipelineResponse,
    RegisterPipelineRequest,
)
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableResponse,
    GetDataInIntervalRequest,
    GetDataInIntervalResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    StartTrainingResponse,
    TrainerAvailableResponse,
    TrainingStatusResponse,
)
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2_grpc import TrainerServerStub


def noop_constructor_mock(self, channel: grpc.Channel) -> None:
    pass


def get_simple_config() -> dict:
    return {
        "storage": {"hostname": "test", "port": 42},
        "trainer_server": {"hostname": "test", "port": 42},
        "selector": {"hostname": "test", "port": 42},
    }


def get_minimal_pipeline_config() -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "training": {
            "gpus": 1,
            "device": "cpu",
            "dataloader_workers": 1,
            "initial_model": "random",
            "initial_pass": {"activated": False},
            "learning_rate": 0.1,
            "batch_size": 42,
            "optimizer": {"name": "SGD"},
            "optimization_criterion": {"name": "CrossEntropyLoss"},
            "checkpointing": {"activated": False},
            "selection_strategy": {"name": "NewDataStrategy"},
        },
        "data": {"dataset_id": "test", "bytes_parser_function": "def bytes_parser_function(x):\n\treturn x"},
        "trigger": {"id": "DataAmountTrigger", "trigger_config": {"data_points_for_trigger": 1}},
    }


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def get_non_connecting_handler(insecure_channel, init) -> GRPCHandler:
    return GRPCHandler(get_simple_config())


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init(test_insecure_channel, test_connection_established):
    handler = GRPCHandler(get_simple_config())

    assert handler.connected_to_storage
    assert handler.connected_to_trainer_server
    assert handler.storage is not None


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_storage(test_insecure_channel, test_connection_established):
    handler = None

    with patch.object(GRPCHandler, "init_storage", return_value=None):
        handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_storage

    handler.init_storage()

    assert handler.connected_to_storage
    assert handler.storage is not None


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=False)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_storage_throws(test_insecure_channel, test_connection_established):
    handler = None

    with patch.object(GRPCHandler, "init_storage", return_value=None):
        with patch.object(GRPCHandler, "init_trainer_server", return_value=None):
            with patch.object(GRPCHandler, "init_selector", return_value=None):
                handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_storage

    with pytest.raises(ConnectionError):
        handler.init_storage()


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_selector(test_insecure_channel, test_connection_established):
    handler = None

    with patch.object(GRPCHandler, "init_selector", return_value=None):
        handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_selector

    handler.init_selector()

    assert handler.connected_to_selector
    assert handler.selector is not None


@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_dataset_available(test_connection_established):
    handler = GRPCHandler(get_simple_config())

    assert handler.storage is not None

    with patch.object(
        handler.storage, "CheckAvailability", return_value=DatasetAvailableResponse(available=True)
    ) as avail_method:
        assert handler.dataset_available("id")
        avail_method.assert_called_once()

    with patch.object(
        handler.storage, "CheckAvailability", return_value=DatasetAvailableResponse(available=False)
    ) as avail_method:
        assert not handler.dataset_available("id")
        avail_method.assert_called_once()


def test_get_new_data_since_throws():
    handler = get_non_connecting_handler()
    handler.connected_to_storage = False
    with pytest.raises(ConnectionError):
        handler.get_new_data_since("dataset_id", 0)


@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_get_new_data_since(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config())

    with patch.object(handler.storage, "GetNewDataSince") as mock:
        mock.return_value = GetNewDataSinceResponse(keys=["test1", "test2"], timestamps=[41, 42], labels=[0, 1])

        result = handler.get_new_data_since("test_dataset", 21)

        assert result == [("test1", 41, 0), ("test2", 42, 1)]
        mock.assert_called_once_with(GetNewDataSinceRequest(dataset_id="test_dataset", timestamp=21))


def test_get_data_in_interval_throws():
    handler = get_non_connecting_handler()
    handler.connected_to_storage = False
    with pytest.raises(ConnectionError):
        handler.get_data_in_interval("dataset_id", 0, 1)


@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_get_data_in_interval(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config())

    with patch.object(handler.storage, "GetDataInInterval") as mock:
        mock.return_value = GetDataInIntervalResponse(keys=["test1", "test2"], timestamps=[41, 42], labels=[0, 1])

        result = handler.get_data_in_interval("test_dataset", 21, 45)

        assert result == [("test1", 41, 0), ("test2", 42, 1)]
        mock.assert_called_once_with(
            GetDataInIntervalRequest(dataset_id="test_dataset", start_timestamp=21, end_timestamp=45)
        )


@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_register_pipeline_at_selector(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config())

    with patch.object(handler.selector, "register_pipeline") as mock:
        mock.return_value = PipelineResponse(pipeline_id=42)

        result = handler.register_pipeline_at_selector(
            {"pipeline": {"name": "test"}, "training": {"dataloader_workers": 2, "selection_strategy": {}}}
        )

        assert result == 42
        mock.assert_called_once_with(RegisterPipelineRequest(num_workers=2, selection_strategy=JsonString(value="{}")))


def test_unregister_pipeline_at_selector():
    handler = get_non_connecting_handler()
    pipeline_id = 42

    # TODO(#64,#124): implement a real test when func is implemented.
    handler.unregister_pipeline_at_selector(pipeline_id)


@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_inform_selector(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config())

    with patch.object(handler.selector, "inform_data") as mock:
        mock.return_value = None

        handler.inform_selector(42, [("a", 42, 0), ("b", 43, 1)])

        mock.assert_called_once_with(
            DataInformRequest(pipeline_id=42, keys=["a", "b"], timestamps=[42, 43], labels=[0, 1])
        )


@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_inform_selector_and_trigger(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config())

    with patch.object(handler.selector, "inform_data_and_trigger") as mock:
        mock.return_value = 12

        assert 12 == handler.inform_selector_and_trigger(42, [("a", 42, 0), ("b", 43, 1)])

        mock.assert_called_once_with(
            DataInformRequest(pipeline_id=42, keys=["a", "b"], timestamps=[42, 43], labels=[0, 1])
        )


@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_trainer_server_available(test_connection_established):
    handler = GRPCHandler(get_simple_config())
    assert handler.trainer_server is not None

    with patch.object(
        handler.trainer_server, "trainer_available", return_value=TrainerAvailableResponse(available=True)
    ) as avail_method:
        assert handler.trainer_server_available()
        avail_method.assert_called_once()


def test_stop_training_at_trainer_server():
    handler = get_non_connecting_handler()
    training_id = 42

    # TODO(#78,#130): implement a real test when func is implemented.
    handler.stop_training_at_trainer_server(training_id)


@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_start_training(test_connection_established):
    handler = GRPCHandler(get_simple_config())
    assert handler.trainer_server is not None

    pipeline_id = 42
    trigger_id = 21
    pipeline_config = get_minimal_pipeline_config()

    with patch.object(
        handler.trainer_server, "start_training", return_value=StartTrainingResponse(training_id=42)
    ) as avail_method:
        assert handler.start_training(pipeline_id, trigger_id, pipeline_config, None) == 42
        avail_method.assert_called_once()


@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_wait_for_training_completion(test_connection_established):
    # This test primarily checks whether we terminate.
    handler = GRPCHandler(get_simple_config())
    assert handler.trainer_server is not None

    with patch.object(
        handler.trainer_server,
        "get_training_status",
        return_value=TrainingStatusResponse(
            valid=True, blocked=False, exception=None, state_available=False, is_running=False
        ),
    ) as avail_method:
        handler.start_training(42)
        avail_method.assert_called_once()
