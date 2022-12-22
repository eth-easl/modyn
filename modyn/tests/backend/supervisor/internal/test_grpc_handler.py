from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from unittest.mock import patch
import grpc

from modyn.storage.storage_pb2_grpc import StorageStub


def noop_constructor_mock(self, channel: grpc.Channel) -> None:  # pylint: disable=unused-argument
    pass


@patch.object(StorageStub, '__init__', noop_constructor_mock)
@patch.object(GRPCHandler, 'connection_established')
@patch.object(grpc, 'insecure_channel')
def test_init(test_insecure_channel, test_connection_established):
    test_insecure_channel.return_value = None
    test_connection_established.return_value = True
    handler = GRPCHandler({'storage': {"hostname": "test", "port": 42}})

    assert handler.connected_to_storage
    assert handler.storage is not None


@patch.object(GRPCHandler,  'init_storage', lambda self: None)
@patch('modyn.backend.supervisor.internal.grpc_handler.TIMEOUT_SEC', 0.5)
def test_connection_established_times_out():
    handler = GRPCHandler({'storage': {"hostname": "test", "port": 42}})
    assert not handler.connection_established(grpc.insecure_channel("1.2.3.4:42"))


@patch('grpc.channel_ready_future')
def test_connection_established_works_mocked(test_channel_ready_future):
    # Pretty dumb test, needs E2E test with running server.

    class MockFuture():
        def result(self, timeout):  # pylint: disable=unused-argument
            return True

    test_channel_ready_future.return_value = MockFuture()
    handler = GRPCHandler({'storage': {"hostname": "test", "port": 42}})
    assert handler.connection_established(grpc.insecure_channel("1.2.3.4:42"))
