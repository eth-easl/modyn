# pylint: disable=unused-argument
"""This tests that the entry point script for the supervisor successfully runs
through.

This is _not_ the place for an integration test.
"""

import os
import pathlib
from unittest.mock import patch

from modyn.supervisor.internal.grpc.supervisor_grpc_server import SupervisorGRPCServer

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

EXAMPLE_SYSTEM_CONFIG = SCRIPT_PATH.parent.parent.parent / "config" / "examples" / "modyn_config.yaml"

NO_FILE = SCRIPT_PATH.parent / "thisshouldnot.exist"


def noop_constructor_mock(self, modyn_config: dict) -> None:
    pass


def noop_enter(self) -> None:
    pass


def noop_exit(self, exc_type, exc_val, exc_tb) -> None:
    pass


@patch.object(SupervisorGRPCServer, "__init__", noop_constructor_mock)
@patch.object(SupervisorGRPCServer, "__enter__", noop_enter)
@patch.object(SupervisorGRPCServer, "__exit__", noop_exit)
def test_supervisor_server_script_runs(script_runner):
    ret = script_runner.run("_modyn_supervisor", str(EXAMPLE_SYSTEM_CONFIG))
    assert ret.success


@patch.object(SupervisorGRPCServer, "__init__", noop_constructor_mock)
@patch.object(SupervisorGRPCServer, "__enter__", noop_enter)
@patch.object(SupervisorGRPCServer, "__exit__", noop_exit)
def test_supervisor_server_fails_on_non_existing_system_config(script_runner):
    ret = script_runner.run("_modyn_supervisor", str(NO_FILE))
    assert not ret.success
