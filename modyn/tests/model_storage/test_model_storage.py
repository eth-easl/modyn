import pathlib
import tempfile
from unittest.mock import patch

from modyn.model_storage import ModelStorage
from modyn.model_storage.internal.grpc.grpc_server import GRPCServer


def get_modyn_config():
    return {"model_storage": {"port": "5001", "ftp_port": "5002"}}


# pylint: disable=unused-argument
def noop_setup_directories(self):
    pass


class MockFTPServer:
    def __init__(self, ftp_port, ftp_directory):  # pylint: disable=unused-argument
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass


class MockGRPCInstance:
    def wait_for_termination(self, *args, **kwargs):  # pylint: disable=unused-argument
        return


class MockGRPCServer(GRPCServer):
    def __enter__(self):
        return MockGRPCInstance()

    def __exit__(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass


@patch.object(ModelStorage, "_setup_model_storage_directories", noop_setup_directories)
def test_model_storage_init():
    model_storage = ModelStorage(get_modyn_config())
    assert model_storage.config == get_modyn_config()


@patch("modyn.model_storage.model_storage.GRPCServer", MockGRPCServer)
@patch("modyn.model_storage.model_storage.FTPServer", MockFTPServer)
@patch("os.makedirs")
def test_cleanup_at_exit(test_os_makedirs):
    ftp_directory = pathlib.Path(tempfile.gettempdir()) / "ftp_model_storage"
    assert not ftp_directory.exists()

    model_storage = ModelStorage(get_modyn_config())
    assert ftp_directory.exists()
    model_storage.run()
    assert not ftp_directory.exists()
