# pylint: disable=unused-argument
""" This tests that the entry point script for the trainer server
successfully runs through. This is _not_ the place for an integration test.
"""
import os
import pathlib
from unittest.mock import patch

from modyn.selector.internal.grpc.selector_server import SelectorGRPCServer

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

EXAMPLE_SYSTEM_CONFIG = SCRIPT_PATH.parent.parent.parent / "config" / "examples" / "modyn_config.yaml"

NO_FILE = SCRIPT_PATH.parent / "thisshouldnot.exist"


def noop_constructor_mock(self, modyn_config: dict) -> None:
    pass


def noop_enter(self) -> None:
    pass


def noop_exit(self, exc_type, exc_val, exc_tb) -> None:
    pass


@patch.object(SelectorGRPCServer, "__init__", noop_constructor_mock)
@patch.object(SelectorGRPCServer, "__enter__", noop_enter)
@patch.object(SelectorGRPCServer, "__exit__", noop_exit)
def test_trainer_server_script_runs(script_runner):
    ret = script_runner.run("_modyn_selector", str(EXAMPLE_SYSTEM_CONFIG))
    assert ret.success


@patch.object(SelectorGRPCServer, "__init__", noop_constructor_mock)
@patch.object(SelectorGRPCServer, "__enter__", noop_enter)
@patch.object(SelectorGRPCServer, "__exit__", noop_exit)
def test_trainer_server_fails_on_non_existing_system_config(script_runner):
    ret = script_runner.run("_modyn_selector", str(NO_FILE))
    assert not ret.success
