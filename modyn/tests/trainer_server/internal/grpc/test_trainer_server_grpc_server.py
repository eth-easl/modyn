# pylint: disable=unused-argument
import tempfile
from unittest.mock import patch

from modyn.trainer_server.internal.grpc.trainer_server_grpc_server import GRPCServer


def get_modyn_config():
    return {
        "trainer_server": {"hostname": "trainer_server", "port": "5001"},
        "storage": {"hostname": "storage", "port": "5002"},
        "selector": {"hostname": "selector", "port": "5003"},
    }


def test_init():
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as tempdir:
        grpc_server = GRPCServer(config, tempdir)
        assert grpc_server.config == config


@patch(
    "modyn.trainer_server.internal.grpc.trainer_server_grpc_server.add_TrainerServerServicer_to_server",
    return_value=None,
)
def test_enter(mock_add_trainer_server_servicer_to_server):
    with tempfile.TemporaryDirectory() as tempdir:
        with GRPCServer(get_modyn_config(), tempdir) as grpc_server:
            assert grpc_server is not None
