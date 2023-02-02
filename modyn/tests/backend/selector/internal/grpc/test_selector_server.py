# pylint: disable=unused-argument,redefined-outer-name
from unittest import mock
from unittest.mock import MagicMock, patch

from modyn.backend.selector.internal.grpc.selector_server import SelectorServer
from modyn.backend.selector.internal.selector_manager import SelectorManager


def noop_init_metadata_db(self):
    pass


def get_modyn_config():
    return {"selector": {"port": "1337"}}


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
def test_init():
    config = get_modyn_config()
    grpc_server = SelectorServer(config)
    assert grpc_server.modyn_config == config


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
def test_prepare_server():
    grpc_server = SelectorServer(get_modyn_config())
    mock_add = mock.Mock()
    grpc_server._add_servicer_to_server_func = mock_add

    assert grpc_server.prepare_server() is not None

    mock_add.assert_called_once()


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
@patch.object(SelectorServer, "prepare_server")
def test_run(test_prepare_server: MagicMock):
    grpc_server = SelectorServer(get_modyn_config())
    mock_start = mock.Mock()
    mock_wait = mock.Mock()

    server = grpc_server.prepare_server()
    server.start = mock_start
    server.wait_for_termination = mock_wait

    test_prepare_server.return_value = server

    grpc_server.run()

    mock_start.assert_called_once()
    mock_wait.assert_called_once()
