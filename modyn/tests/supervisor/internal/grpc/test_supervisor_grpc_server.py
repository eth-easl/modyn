# pylint: disable=unused-argument,redefined-outer-name
from unittest.mock import Mock, patch

from modyn.common.grpc import GenericGRPCServer
from modyn.supervisor.internal.grpc.supervisor_grpc_server import SupervisorGRPCServer
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.supervisor import Supervisor


def noop_init_metadata_db(self):
    pass


def noop(self) -> None:
    pass


def get_minimal_modyn_config():
    return {"supervisor": {"hostname": "supervisor", "port": 42}}


def noop_constructor_mock(self, modyn_config: dict) -> None:
    pass


@patch.object(GRPCHandler, "__init__", noop_constructor_mock)
@patch.object(Supervisor, "init_cluster_connection", noop)
@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
@patch.object(GenericGRPCServer, "__init__")
def test_init(generic_grpc_server_init_mock: Mock):
    modyn_config = get_minimal_modyn_config()
    grpc_server = SupervisorGRPCServer(modyn_config)
    assert grpc_server.modyn_config == modyn_config
    assert isinstance(grpc_server.supervisor, Supervisor)

    expected_callback_kwargs = {"supervisor": grpc_server.supervisor}
    generic_grpc_server_init_mock.assert_called_once_with(
        modyn_config, modyn_config["supervisor"]["port"], SupervisorGRPCServer.callback, expected_callback_kwargs
    )
