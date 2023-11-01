# pylint: disable=unused-argument,redefined-outer-name
from unittest.mock import patch

from modyn.supervisor.internal.grpc.supervisor_grpc_server import SupervisorGRPCServer
from modyn.supervisor.supervisor import Supervisor


def noop_init_metadata_db(self):
    pass


def get_modyn_config():
    return {"supervisor": {"port": "50063"}}


@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
def test_init():
    modyn_config = get_modyn_config()
    grpc_server = SupervisorGRPCServer(modyn_config)
    assert grpc_server.modyn_config == modyn_config
    assert isinstance(grpc_server.supervisor, Supervisor)
