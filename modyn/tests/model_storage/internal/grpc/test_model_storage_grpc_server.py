# pylint: disable=unused-argument
import pathlib
from unittest.mock import patch

from modyn.model_storage.internal.grpc.grpc_server import GRPCServer


def get_modyn_config():
    return {"model_storage": {"port": "50051"}}


def test_init():
    grpc_server = GRPCServer(get_modyn_config(), pathlib.Path.cwd() / "temp_dir")
    assert grpc_server.modyn_config == get_modyn_config()
    assert str(grpc_server.storage_dir) == str(pathlib.Path.cwd() / "temp_dir")


@patch("modyn.model_storage.internal.grpc.grpc_server.add_ModelStorageServicer_to_server", return_value=None)
def test_enter(mock_add_model_storage_servicer_to_server):
    with GRPCServer(get_modyn_config(), pathlib.Path.cwd() / "temp_dir") as grpc_server:
        assert grpc_server is not None
