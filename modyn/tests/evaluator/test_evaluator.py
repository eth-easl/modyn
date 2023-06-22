import os
import pathlib
import tempfile
from unittest.mock import patch

from modyn.evaluator import Evaluator
from modyn.evaluator.internal.grpc.evaluator_grpc_server import EvaluatorGRPCServer

modyn_config = (
    pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "examples" / "modyn_config.yaml"
)


class MockGRPCInstance:
    def wait_for_termination(self, *args, **kwargs):  # pylint: disable=unused-argument
        return


class MockGRPCServer(EvaluatorGRPCServer):
    def __enter__(self):
        return MockGRPCInstance()

    def __exit__(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass


def test_init():
    evaluator = Evaluator(modyn_config)
    assert evaluator.config == modyn_config


@patch("modyn.evaluator.evaluator.EvaluatorGRPCServer", MockGRPCServer)
def test_run():
    trainer_server = Evaluator(modyn_config)
    trainer_server.run()


@patch("modyn.evaluator.evaluator.EvaluatorGRPCServer", MockGRPCServer)
def test_cleanup_at_exit():
    modyn_dir = pathlib.Path(tempfile.gettempdir()) / "modyn_evaluator"
    assert not modyn_dir.exists()

    evaluator = Evaluator(modyn_config)
    assert modyn_dir.exists()
    evaluator.run()
    assert not modyn_dir.exists()
