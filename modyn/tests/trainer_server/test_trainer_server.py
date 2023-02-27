import pathlib
import tempfile
from unittest.mock import patch

from modyn.trainer_server.internal.grpc.trainer_server_grpc_server import GRPCServer
from modyn.trainer_server.trainer_server import TrainerServer


class MockGRPCInstance:
    def wait_for_termination(self, *args, **kwargs):  # pylint: disable=unused-argument
        return


class MockGRPCServer(GRPCServer):
    def __enter__(self):
        return MockGRPCInstance()

    def __exit__(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass


def get_modyn_config():
    return {"trainer_server": {"port": "5001", "type": "grpc", "ftp_port": "5002"}}


def test_init():
    config = get_modyn_config()
    trainer_server = TrainerServer(config)
    assert trainer_server.config == config


@patch("modyn.trainer_server.trainer_server.GRPCServer", MockGRPCServer)
@patch("modyn.trainer_server.trainer_server.FTPServer", MockGRPCServer)
def test_run():
    config = get_modyn_config()
    trainer_server = TrainerServer(config)
    trainer_server.run()


@patch("modyn.trainer_server.trainer_server.GRPCServer", MockGRPCServer)
@patch("modyn.trainer_server.trainer_server.FTPServer", MockGRPCServer)
def test_cleanup_at_exit():
    modyn_dir = pathlib.Path(tempfile.gettempdir()) / "modyn"
    assert not modyn_dir.exists()

    trainer_server = TrainerServer(get_modyn_config())
    assert modyn_dir.exists()
    trainer_server.run()
    assert not modyn_dir.exists()
