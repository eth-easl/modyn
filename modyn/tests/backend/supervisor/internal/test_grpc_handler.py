# pylint: disable=unused-argument,no-value-for-parameter
from unittest.mock import patch

import grpc
import pytest
from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler

# pylint: disable-next=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableResponse,
    GetDataInIntervalRequest,
    GetDataInIntervalResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub


def noop_constructor_mock(self, channel: grpc.Channel) -> None:
    pass


def get_simple_config() -> dict:
    return {"storage": {"hostname": "test", "port": 42}}


@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def get_non_connecting_handler(insecure_channel, init) -> GRPCHandler:
    return GRPCHandler(get_simple_config())


@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init(test_insecure_channel, test_connection_established):
    handler = GRPCHandler(get_simple_config())

    assert handler.connected_to_storage
    assert handler.storage is not None


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


@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch("modyn.backend.supervisor.internal.grpc_handler.grpc_connection_established", return_value=False)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_storage_throws(test_insecure_channel, test_connection_established):
    handler = None

    with patch.object(GRPCHandler, "init_storage", return_value=None):
        handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_storage

    with pytest.raises(ConnectionError):
        handler.init_storage()


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
        # TODO(#76): The storage should return the timestamp as well
        mock.return_value = GetNewDataSinceResponse(keys=["test1", "test2"])

        result = handler.get_new_data_since("test_dataset", 21)

        assert result == [("test1", 42), ("test2", 42)]  # timestamp is currently hardcoded
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
        # TODO(#76): The storage should return the timestamp as well
        mock.return_value = GetDataInIntervalResponse(keys=["test1", "test2"])

        result = handler.get_data_in_interval("test_dataset", 21, 45)

        assert result == [("test1", 42), ("test2", 42)]  # timestamp is currently hardcoded
        mock.assert_called_once_with(
            GetDataInIntervalRequest(dataset_id="test_dataset", start_timestamp=21, end_timestamp=45)
        )


def test_register_pipeline_at_selector():
    handler = get_non_connecting_handler()
    pipeline_config = {}

    # TODO(#64): implement a real test when func is implemented.
    assert handler.register_pipeline_at_selector(pipeline_config) == 42


def test_unregister_pipeline_at_selector():
    handler = get_non_connecting_handler()
    pipeline_id = 42

    # TODO(#64): implement a real test when func is implemented.
    handler.unregister_pipeline_at_selector(pipeline_id)


def test_inform_selector():
    handler = get_non_connecting_handler()
    pipeline_id = 42
    data = []

    # TODO(#64): implement a real test when func is implemented.
    handler.inform_selector(pipeline_id, data)


def test_inform_selector_and_trigger():
    handler = get_non_connecting_handler()
    pipeline_id = 42
    data = []

    # TODO(#64): implement a real test when func is implemented.
    handler.inform_selector_and_trigger(pipeline_id, data)


def test_trainer_server_available():
    handler = get_non_connecting_handler()

    # TODO(#78): implement a real test when func is implemented.
    assert handler.trainer_server_available()


def test_shutdown_trainer_server():
    handler = get_non_connecting_handler()
    training_id = 42

    # TODO(#78): implement a real test when func is implemented.
    handler.shutdown_trainer_server(training_id)


def test_start_trainer_server():
    handler = get_non_connecting_handler()
    pipeline_id = 42
    trigger_id = 21
    pipeline_config = {}

    # TODO(#78): implement a real test when func is implemented.
    assert handler.start_trainer_server(pipeline_id, trigger_id, pipeline_config) == 42


def test_wait_for_training_completion():
    handler = get_non_connecting_handler()
    training_id = 42

    # TODO(#78): implement a real test when func is implemented.
    handler.wait_for_training_completion(training_id)
