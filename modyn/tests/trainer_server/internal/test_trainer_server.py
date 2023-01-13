from unittest.mock import patch
from modyn.trainer_server.internal.grpc.grpc_server import GRPCServer
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
    return {"trainer": {"port": "5001", "type": "grpc"}}


def test_init():
    config = get_modyn_config()
    trainer_server = TrainerServer(config)
    assert trainer_server.config == config


@patch("modyn.trainer_server.trainer_server.GRPCServer", MockGRPCServer)
def test_run():
    config = get_modyn_config()
    trainer_server = TrainerServer(config)
    trainer_server.run()
