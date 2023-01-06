# pylint: disable=unused-argument
from modyn.backend.selector.internal.grpc.grpc_handler import GRPCHandler
import modyn.utils
from unittest.mock import patch
import grpc
import pytest

from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2_grpc import MetadataStub
# # pylint: disable-next=no-name-in-module
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2 import GetResponse, RegisterResponse, TrainingResponse  # noqa: E501, E402


def noop_constructor_mock(self, channel: grpc.Channel) -> None:  # pylint: disable=unused-argument
    pass


def get_simple_config() -> dict:
    return {'metadata_database': {"hostname": "test", "port": 42}}


@patch.object(MetadataStub, '__init__', noop_constructor_mock)
@patch.object(grpc, 'insecure_channel', return_value=None)
@patch.object(modyn.utils, 'connection_established', return_value=True)
def test_init(test_insecure_channel, test__connection_established):
    test__connection_established.return_value = True

    handler = GRPCHandler(get_simple_config())

    assert handler.connected_to_metadata
    assert handler.metadata_database is not None


def test_connection_established_times_out():
    assert not modyn.utils.connection_established(grpc.insecure_channel("1.2.3.4:42"), 0.5)


@patch('grpc.channel_ready_future')
def test_connection_established_works_mocked(test_channel_ready_future):
    # Pretty dumb test, needs E2E test with running server.

    class MockFuture():
        def result(self, timeout):  # pylint: disable=unused-argument
            return True

    test_channel_ready_future.return_value = MockFuture()
    assert modyn.utils.connection_established(grpc.insecure_channel("1.2.3.4:42"))


@patch.object(MetadataStub, '__init__', noop_constructor_mock)
@patch.object(modyn.utils, 'connection_established', return_value=True)
@patch.object(grpc, 'insecure_channel', return_value=None)
def test_init_metadata(test_insecure_channel, test__connection_established):
    handler = None

    with patch.object(GRPCHandler, '_init_metadata', return_value=None):
        handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_metadata

    handler._init_metadata()

    assert handler.connected_to_metadata
    assert handler.metadata_database is not None


@patch.object(MetadataStub, '__init__', noop_constructor_mock)
@patch.object(modyn.utils, 'connection_established', return_value=False)
@patch.object(grpc, 'insecure_channel', return_value=None)
def test_init_metadata_throws(test_insecure_channel, test__connection_established):
    handler = None

    with patch.object(GRPCHandler, '_init_metadata', return_value=None):
        handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_metadata

    with pytest.raises(ConnectionError):
        handler._init_metadata()


@patch.object(modyn.utils, 'connection_established', return_value=True)
def test_register_training(test__connection_established):
    handler = GRPCHandler(get_simple_config())

    assert handler.metadata_database is not None

    with patch.object(handler.metadata_database, 'RegisterTraining',
                      return_value=RegisterResponse(training_id=1)) as avail_method:
        assert handler.register_training(1000, 10) == 1
        avail_method.assert_called_once()


@patch.object(modyn.utils, 'connection_established', return_value=True)
def test_get_info_for_training(test__connection_established):
    handler = GRPCHandler(get_simple_config())

    assert handler.metadata_database is not None

    with patch.object(handler.metadata_database, 'GetTrainingInfo',
                      return_value=TrainingResponse(training_set_size=1000, num_workers=10)) as avail_method:
        assert handler.get_info_for_training(0) == (1000, 10)
        avail_method.assert_called_once()


@patch.object(modyn.utils, 'connection_established', return_value=True)
def test_get_samples_by_metadata_query(test__connection_established):
    handler = GRPCHandler(get_simple_config())

    assert handler.metadata_database is not None

    keys = ['a', 'b']
    scores = [.25, .5]
    data = ['a', 'b']
    seen = [1, 0]
    label = [5, 6]

    with patch.object(handler.metadata_database, 'GetByQuery',
                      return_value=GetResponse(keys=keys,
                                               scores=scores,
                                               data=data,
                                               seen=seen,
                                               label=label)) as avail_method:
        got = handler.get_samples_by_metadata_query('sample_query')
        expect = (keys, scores, seen, label, data)
        assert got == expect
        avail_method.assert_called_once()
