from unittest.mock import patch

from modyn.storage.internal.grpc.grpc_server import GRPCServer


def get_modyn_config():
    return {
        'storage': {
            'port': '50051',
            'type': 'grpc'
        }
    }


def test_init():
    grpc_server = GRPCServer(get_modyn_config())
    assert grpc_server.modyn_config == get_modyn_config()


@patch.object(GRPCServer, '_add_storage_servicer_to_server', return_value=None)
def test_enter(mock_add_storage_servicer_to_server):
    with GRPCServer(get_modyn_config()) as grpc_server:
        assert grpc_server is not None
        mock_add_storage_servicer_to_server.assert_called_once()
