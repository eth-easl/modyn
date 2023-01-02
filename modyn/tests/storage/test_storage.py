import os
from unittest.mock import patch
import pytest

from modyn.storage.storage import Storage
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.grpc.grpc_server import GRPCServer
from modyn.storage.internal.seeker import Seeker


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
                'database': os.path.join(os.getcwd(), 'modyn', 'tests', 'storage', 'test_storage.db')
            },
            'seeker': {
                'interval': 1
            },
            'datasets': [
                {
                    'name': 'test',
                    'base_path': '/tmp/modyn',
                    'filesystem_wrapper_type': 'LocalFilesystemWrapper',
                    'file_wrapper_type': 'MNISTWebdatasetFileWrapper',
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
            'type': 'MNIST',
            'path': '/tmp/modyn'
        },
        'odm': {
            'type': 'MNIST'
        }
    }


def teardown():
    os.remove(os.path.join(os.getcwd(), 'modyn', 'tests', 'storage', 'test_storage.db'))


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
    modyn_config = os.path.join(os.getcwd(), 'modyn', 'config', 'modyn_config.yaml')
    storage = Storage(modyn_config)
    assert storage.modyn_config == modyn_config


def test_validate_config():
    modyn_config = os.path.join(os.getcwd(), 'modyn', 'config', 'modyn_config.yaml')
    storage = Storage(modyn_config)
    assert storage._validate_config()[0]


@patch.object(Seeker, 'run', lambda *args, **kwargs: None)
@patch('modyn.storage.storage.GRPCServer', MockGRPCServer)
def test_run():
    with DatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_all()
    storage = Storage(get_minimal_modyn_config())
    storage.run()


def test_invalid_config():
    with pytest.raises(ValueError):
        Storage(get_invalid_modyn_config())
