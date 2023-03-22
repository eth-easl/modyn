from unittest import mock
from unittest.mock import MagicMock, patch

from modyn.metadata_processor.internal.grpc.metadata_processor_server import MetadataProcessorServer


def get_modyn_config():
    return {"metadata_processor": {"port": "1336"}}


def test_constructor():
    config = get_modyn_config()
    grpc_server = MetadataProcessorServer(get_modyn_config())
    assert grpc_server.config == config


def test_prepare_server():
    grpc_server = MetadataProcessorServer(get_modyn_config())
    mock_add = mock.Mock()
    grpc_server._add_servicer_to_server_func = mock_add

    assert grpc_server.prepare_server() is not None

    mock_add.assert_called_once()


@patch.object(MetadataProcessorServer, "prepare_server")
def test_run(test__prepare_server: MagicMock):
    grpc_server = MetadataProcessorServer(get_modyn_config())
    mock_start = mock.Mock()
    mock_wait = mock.Mock()

    server = grpc_server.prepare_server()
    server.start = mock_start
    server.wait_for_termination = mock_wait

    test__prepare_server.return_value = server

    grpc_server.run()

    mock_start.assert_called_once()
    mock_wait.assert_called_once()
