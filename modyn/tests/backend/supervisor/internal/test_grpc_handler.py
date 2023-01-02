# pylint: disable=unused-argument
from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from unittest.mock import patch
import grpc
import pytest

from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
# pylint: disable-next=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2 import DatasetAvailableResponse


def noop_constructor_mock(self, channel: grpc.Channel) -> None:
    pass


def get_simple_config() -> dict:
    return {'storage': {"hostname": "test", "port": 42}}


@patch.object(StorageStub, '__init__', noop_constructor_mock)
@patch.object(GRPCHandler, 'connection_established', return_value=True)
@patch.object(grpc, 'insecure_channel', return_value=None)
def test_init(test_insecure_channel, test_connection_established):
    handler = GRPCHandler(get_simple_config())

    assert handler.connected_to_storage
    assert handler.storage is not None


@patch.object(GRPCHandler, 'init_storage', lambda self: None)
@patch('modyn.backend.supervisor.internal.grpc_handler.TIMEOUT_SEC', 0.5)
def test_connection_established_times_out():
    handler = GRPCHandler(get_simple_config())
    assert not handler.connection_established(grpc.insecure_channel("1.2.3.4:42"))


@patch('grpc.channel_ready_future')
def test_connection_established_works_mocked(test_channel_ready_future):
    # Pretty dumb test, needs E2E test with running server.

    class MockFuture():
        def result(self, timeout):
            return True

    test_channel_ready_future.return_value = MockFuture()
    handler = GRPCHandler(get_simple_config())
    assert handler.connection_established(grpc.insecure_channel("1.2.3.4:42"))


@patch.object(StorageStub, '__init__', noop_constructor_mock)
@patch.object(GRPCHandler, 'connection_established', return_value=True)
@patch.object(grpc, 'insecure_channel', return_value=None)
def test_init_storage(test_insecure_channel, test_connection_established):
    handler = None

    with patch.object(GRPCHandler, 'init_storage', return_value=None):
        handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_storage

    handler.init_storage()

    assert handler.connected_to_storage
    assert handler.storage is not None


@patch.object(StorageStub, '__init__', noop_constructor_mock)
@patch.object(GRPCHandler, 'connection_established', return_value=False)
@patch.object(grpc, 'insecure_channel', return_value=None)
def test_init_storage_throws(test_insecure_channel, test_connection_established):
    handler = None

    with patch.object(GRPCHandler, 'init_storage', return_value=None):
        handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_storage

    with pytest.raises(ConnectionError):
        handler.init_storage()


@patch.object(GRPCHandler, 'connection_established', return_value=True)
def test_dataset_available(test_connection_established):
    handler = GRPCHandler(get_simple_config())

    assert handler.storage is not None

    with patch.object(handler.storage, 'CheckAvailability',
                      return_value=DatasetAvailableResponse(available=True)) as avail_method:
        assert handler.dataset_available("id")
        avail_method.assert_called_once()

    with patch.object(handler.storage, 'CheckAvailability',
                      return_value=DatasetAvailableResponse(available=False)) as avail_method:
        assert not handler.dataset_available("id")
        avail_method.assert_called_once()
