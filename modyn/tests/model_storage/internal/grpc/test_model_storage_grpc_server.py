# pylint: disable=unused-argument
import pathlib
from unittest.mock import patch

from modyn.model_storage.internal import ModelStorageManager
from modyn.model_storage.internal.grpc.grpc_server import GRPCServer


def get_modyn_config():
    return {"model_storage": {"port": "50051"}}


def test_init():
    grpc_server = GRPCServer(get_modyn_config(), pathlib.Path.cwd() / "storage_dir", pathlib.Path.cwd() / "ftp_dir")
    assert grpc_server.modyn_config == get_modyn_config()
    assert str(grpc_server.storage_dir) == str(pathlib.Path.cwd() / "storage_dir")
    assert str(grpc_server.ftp_directory) == str(pathlib.Path.cwd() / "ftp_dir")


@patch("modyn.model_storage.internal.grpc.grpc_server.add_ModelStorageServicer_to_server", return_value=None)
@patch.object(ModelStorageManager, "__init__", return_value=None)
def test_enter(mock_init_model_storage_manager, mock_add_model_storage_servicer_to_server):
    with GRPCServer(
        get_modyn_config(), pathlib.Path.cwd() / "storage_dir", pathlib.Path.cwd() / "ftp_dir"
    ) as grpc_server:
        assert grpc_server is not None
