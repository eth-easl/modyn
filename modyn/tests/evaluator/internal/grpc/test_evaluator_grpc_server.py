# pylint: disable=unused-argument
import pathlib
import tempfile
from unittest.mock import patch

from modyn.evaluator.internal.grpc.evaluator_grpc_server import EvaluatorGRPCServer
from modyn.evaluator.internal.grpc.evaluator_grpc_servicer import EvaluatorGRPCServicer


def noop_constructor_mock(self, config, tempdir) -> None:
    pass


def get_modyn_config():
    return {
        "evaluator": {"hostname": "evaluator", "port": "6000"},
        "model_storage": {"hostname": "model_storage", "port": "6001", "ftp_port": "6010"},
        "storage": {"hostname": "storage", "port": "6002"},
    }


def test_init():
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as tempdir:
        grpc_server = EvaluatorGRPCServer(config, tempdir)
        assert grpc_server.modyn_config == config


@patch.object(EvaluatorGRPCServicer, "__init__", noop_constructor_mock)
@patch(
    "modyn.evaluator.internal.grpc.evaluator_grpc_server.add_EvaluatorServicer_to_server",
    return_value=None,
)
def test_enter(mock_add_evaluator_servicer_to_server):
    with tempfile.TemporaryDirectory() as tempdir:
        with EvaluatorGRPCServer(get_modyn_config(), pathlib.Path(tempdir)) as grpc_server:
            assert grpc_server is not None
