# pylint: disable=unused-argument
import os
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
    grpc_server = GRPCServer(config)
    assert grpc_server.config == config


@patch(
    "modyn.trainer_server.internal.grpc.trainer_server_grpc_server.add_TrainerServerServicer_to_server",
    return_value=None,
)
def test_enter(mock_add_trainer_server_servicer_to_server):
    with GRPCServer(get_modyn_config()) as grpc_server:
        assert grpc_server is not None


@patch(
    "modyn.trainer_server.internal.grpc.trainer_server_grpc_server.add_TrainerServerServicer_to_server",
    return_value=None,
)
def test_cleanup_at_exit(mock_add_trainer_server_servicer_to_server):
    with GRPCServer(get_modyn_config()) as _:
        with open(f"{tempfile.gettempdir()}/training_0", "w", encoding="utf-8") as _:
            assert os.path.isfile(f"{tempfile.gettempdir()}/training_0")

    assert not os.path.isfile(f"{tempfile.gettempdir()}/training_0")
