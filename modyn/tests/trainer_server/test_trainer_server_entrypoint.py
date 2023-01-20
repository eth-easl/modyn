# pylint: disable=unused-argument
""" This tests that the entry point script for the trainer server
successfully runs through. This is _not_ the place for an integration test.
"""
import os
import pathlib
from unittest.mock import patch

from modyn.trainer_server.trainer_server import TrainerServer


SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

EXAMPLE_SYSTEM_CONFIG = SCRIPT_PATH.parent.parent.parent / "config" / "examples" / "modyn_config.yaml"

NO_FILE = SCRIPT_PATH.parent / "thisshouldnot.exist"


def noop_constructor_mock(self, modyn_config: dict) -> None:
    pass


def noop_run(self) -> None:
    pass


@patch.object(TrainerServer, "__init__", noop_constructor_mock)
@patch.object(TrainerServer, "run", noop_run)
def test_trainer_server_script_runs(script_runner):
    ret = script_runner.run("_modyn_trainer_server", str(EXAMPLE_SYSTEM_CONFIG))
    assert ret.success


@patch.object(TrainerServer, "__init__", noop_constructor_mock)
@patch.object(TrainerServer, "run", noop_run)
def test_trainer_server_fails_on_non_existing_system_config(script_runner):
    ret = script_runner.run("_modyn_trainer_server", str(NO_FILE))
    assert not ret.success
