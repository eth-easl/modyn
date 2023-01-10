# pylint: disable=unused-argument,redefined-outer-name
from modyn.utils import grpc_connection_established

from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from unittest.mock import patch
import grpc


@patch.object(GRPCHandler, 'init_storage', lambda self: None)
def test_connection_established_times_out():
    assert not grpc_connection_established(grpc.insecure_channel("1.2.3.4:42"), 0.5)


@patch('grpc.channel_ready_future')
def test_connection_established_works_mocked(test_channel_ready_future):
    # Pretty dumb test, needs E2E test with running server.

    class MockFuture():
        def result(self, timeout):
            return True

    test_channel_ready_future.return_value = MockFuture()
    assert grpc_connection_established(grpc.insecure_channel("1.2.3.4:42"))
