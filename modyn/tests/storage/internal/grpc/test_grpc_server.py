# pylint: disable=unused-argument
from unittest.mock import patch

from modyn.storage.internal.grpc.grpc_server import StorageGRPCServer


def get_modyn_config():
    return {"storage": {"port": "50051", "type": "grpc", "sample_batch_size": 1024}}


def test_init():
    grpc_server = StorageGRPCServer(get_modyn_config())
    assert grpc_server.modyn_config == get_modyn_config()


@patch("modyn.storage.internal.grpc.grpc_server.add_StorageServicer_to_server", return_value=None)
def test_enter(mock_add_storage_servicer_to_server):
    with StorageGRPCServer(get_modyn_config()) as grpc_server:
        assert grpc_server is not None
