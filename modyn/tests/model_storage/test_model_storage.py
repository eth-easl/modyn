import os
import pathlib
from unittest.mock import patch

import pytest
from modyn.model_storage import ModelStorage
from modyn.model_storage.internal.grpc.grpc_server import GRPCServer

modyn_config = (
    pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "examples" / "modyn_config.yaml"
)


def get_invalid_modyn_config() -> dict:
    return {"invalid": "not_valid"}


# pylint: disable=unused-argument
def noop_setup_directory(self):
    pass


class MockFTPServer:
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


@patch.object(ModelStorage, "_setup_model_storage_directory", noop_setup_directory)
def test_model_storage_init():
    model_storage = ModelStorage(modyn_config)
    assert model_storage.config == modyn_config


@patch.object(ModelStorage, "_setup_model_storage_directory", noop_setup_directory)
def test_validate_config():
    model_storage = ModelStorage(modyn_config)
    assert model_storage._validate_config()[0]


def test_invalid_config():
    with pytest.raises(ValueError):
        ModelStorage(get_invalid_modyn_config())
