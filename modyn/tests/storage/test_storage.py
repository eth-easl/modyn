import os
import pathlib
from unittest.mock import patch

import pytest
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.grpc.grpc_server import GRPCServer
from modyn.storage.storage import Storage

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"
modyn_config = (
    pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "examples" / "modyn_config.yaml"
)


def get_minimal_modyn_config() -> dict:
    return {
        "storage": {
            "port": "50051",
            "hostname": "localhost",
            "sample_batch_size": 1024,
            "filesystem": {"type": "LocalFilesystemWrapper", "base_path": "/tmp/modyn"},
            "database": {
                "drivername": "sqlite",
                "username": "",
                "password": "",
                "host": "",
                "port": "0",
                "database": f"{database_path}",
            },
            "new_file_watcher": {"interval": 1},
            "datasets": [
                {
                    "name": "test",
                    "base_path": "/tmp/modyn",
                    "filesystem_wrapper_type": "LocalFilesystemWrapper",
                    "file_wrapper_type": "SingleSampleFileWrapper",
                    "description": "test",
                    "version": "0.0.1",
                    "file_wrapper_config": {},
                }
            ],
        },
        "project": {"name": "test", "version": "0.0.1"},
        "input": {"type": "LOCAL", "path": "/tmp/modyn"},
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "host": "",
            "port": "0",
            "database": f"{database_path}",
        },
        "selector": {"hostname": "host", "port": "1337"},
        "trainer_server": {"hostname": "host", "port": "1337"},
    }


def teardown():
    os.remove(database_path)


def setup():
    if database_path.exists():
        os.remove(database_path)
    os.makedirs(database_path.parent, exist_ok=True)


def get_invalid_modyn_config() -> dict:
    return {"invalid": "invalid"}


class MockGRPCInstance:
    def wait_for_termination(self, *args, **kwargs):  # pylint: disable=unused-argument
        return


class MockGRPCServer(GRPCServer):
    def __enter__(self):
        return MockGRPCInstance()

    def __exit__(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass


def test_storage_init():
    storage = Storage(modyn_config)
    assert storage.modyn_config == modyn_config


def test_validate_config():
    storage = Storage(modyn_config)
    assert storage._validate_config()[0]


@patch("modyn.storage.storage.GRPCServer", MockGRPCServer)
def test_run():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    storage = Storage(get_minimal_modyn_config())
    storage.run()


def test_invalid_config():
    with pytest.raises(ValueError):
        Storage(get_invalid_modyn_config())
