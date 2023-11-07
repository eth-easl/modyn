# pylint: disable=unused-argument,redefined-outer-name
from unittest.mock import patch

from modyn.supervisor.internal.grpc.supervisor_grpc_server import SupervisorGRPCServer
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.supervisor import Supervisor


def noop_init_metadata_db(self):
    pass


def noop(self) -> None:
    pass
\

def get_minimal_modyn_config():
    return {"supervisor": {"hostname": "supervisor", "port": 42}}


def noop_constructor_mock(self, modyn_config: dict) -> None:
    pass


@patch.object(GRPCHandler, "__init__", noop_constructor_mock)
@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
@patch.object(Supervisor, "monitor_pipelines", noop)
def test_init():
    modyn_config = get_minimal_modyn_config()
    grpc_server = SupervisorGRPCServer(modyn_config)
    assert grpc_server.modyn_config == modyn_config
    assert isinstance(grpc_server.supervisor, Supervisor)
