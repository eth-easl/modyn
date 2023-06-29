# pylint: disable=unused-argument
import tempfile
from unittest.mock import patch

from modyn.trainer_server.internal.grpc.trainer_server_grpc_server import GRPCServer
from modyn.trainer_server.internal.grpc.trainer_server_grpc_servicer import TrainerServerGRPCServicer


def get_modyn_config():
    return {
        "trainer_server": {
            "hostname": "trainer_server",
            "port": "5001",
            "offline_dataset_directory": "/tmp/offline_dataset",
        },
        "storage": {"hostname": "storage", "port": "5002"},
        "selector": {"hostname": "selector", "port": "5003"},
        "model_storage": {"hostname": "model_storage", "port": "5004"},
    }


def test_init():
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as tempdir:
        grpc_server = GRPCServer(config, tempdir)
        assert grpc_server.config == config


@patch.object(TrainerServerGRPCServicer, "connect_to_model_storage", return_value=None)
@patch(
    "modyn.trainer_server.internal.grpc.trainer_server_grpc_server.add_TrainerServerServicer_to_server",
    return_value=None,
)
def test_enter(mock_add_trainer_server_servicer_to_server, mock_connect_to_model_storage):
    with tempfile.TemporaryDirectory() as tempdir:
        with GRPCServer(get_modyn_config(), tempdir) as grpc_server:
            assert grpc_server is not None
