from modyn.backend.supervisor import Supervisor
from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from unittest.mock import patch


def get_minimal_pipeline_config() -> dict:
    return {'pipeline': {
        'name': 'Test'
    },
        'model': {'id': 'ResNet18'},
        'training': {'gpus': 1},
        'data': {'dataset_id': 'test'}}


def get_minimal_system_config() -> dict:
    return {}


@patch.object(GRPCHandler, 'init_storage')
@patch.object(GRPCHandler, 'connection_established')
@patch.object(GRPCHandler, 'dataset_available')
def test_initialization(test_dataset_available,  test_connection_established, test_init_storage) -> None:
    test_init_storage.return_value = None
    test_connection_established.return_value = True
    test_dataset_available.return_value = True

    test = Supervisor(get_minimal_pipeline_config(),  # noqa: F841 # pylint: disable=unused-variable
                      get_minimal_system_config(), None)
