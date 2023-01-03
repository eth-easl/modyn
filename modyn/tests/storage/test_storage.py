import os
from unittest.mock import patch
import pytest
import pathlib

from modyn.storage.storage import Storage
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.grpc.grpc_server import GRPCServer
from modyn.storage.internal.dataset_watcher import DatasetWatcher


database_path = pathlib.Path(os.path.abspath(__file__)).parent / 'test_storage.db'
modyn_config = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent \
                / "config" / "examples" / "modyn_config.yaml"


def get_minimal_modyn_config() -> dict:
    return {
        'storage': {
            'filesystem': {
                'type': 'LocalFilesystemWrapper',
                'base_path': '/tmp/modyn'
            },
            'database': {
                'drivername': 'sqlite',
                'username': '',
                'password': '',
                'host': '',
                'port': '0',
                'database': f'{database_path}'
            },
            'dataset_watcher': {
                'interval': 1
            },
            'datasets': [
                {
                    'name': 'test',
                    'base_path': '/tmp/modyn',
                    'filesystem_wrapper_type': 'LocalFilesystemWrapper',
                    'file_wrapper_type': 'WebdatasetFileWrapper',
                    'description': 'test',
                    'version': '0.0.1',
                }
            ]
        },
        'project': {
            'name': 'test',
            'version': '0.0.1'
        },
        'input': {
            'type': 'LOCAL',
            'path': '/tmp/modyn'
        },
        'odm': {
            'type': 'LOCAL'
        }
    }


def teardown():
    os.remove(database_path)


def get_invalid_modyn_config() -> dict:
    return {
        'invalid': 'invalid'
    }


class MockGRPCInstance():
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


@patch.object(DatasetWatcher, 'run', lambda *args, **kwargs: None)
@patch('modyn.storage.storage.GRPCServer', MockGRPCServer)
def test_run():
    with DatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_all()
    storage = Storage(get_minimal_modyn_config())
    storage.run()


def test_invalid_config():
    with pytest.raises(ValueError):
        Storage(get_invalid_modyn_config())
