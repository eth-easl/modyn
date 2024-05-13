# pylint: disable=unused-argument
""" This tests that the entry point script for the supervisor
successfully runs through. This is _not_ the place for an integration test.
"""
import os
import pathlib
from typing import Optional
from unittest.mock import patch

from modynclient.client import Client

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

EXAMPLE_PIPELINE = SCRIPT_PATH.parent.parent.parent / "config" / "examples" / "mnist.yaml"
EXAMPLE_SYSTEM_CONFIG = SCRIPT_PATH.parent.parent.parent / "config" / "examples" / "modyn_client_config.yaml"

NO_FILE = SCRIPT_PATH.parent / "thisshouldnot.exist"


def noop_constructor_mock(
    self,
    client_config: dict,
    pipeline_config: dict,
    start_replay_at: Optional[int] = None,
    stop_replay_at: Optional[int] = None,
    maximum_triggers: Optional[int] = None,
) -> None:
    pass


def noop_poll_pipeline_status(self) -> None:
    pass


@patch.object(Client, "__init__", noop_constructor_mock)
@patch.object(Client, "start_pipeline", return_value=True)
@patch.object(Client, "poll_pipeline_status", noop_poll_pipeline_status)
def test_modyn_client_script_runs(script_runner):
    ret = script_runner.run("_modyn_client", str(EXAMPLE_PIPELINE), str(EXAMPLE_SYSTEM_CONFIG))
    assert ret.success


@patch.object(Client, "__init__", noop_constructor_mock)
def test_supervisor_script_fails_on_non_existing_system_config(script_runner):
    assert not NO_FILE.is_file(), "File that shouldn't exist exists."
    ret = script_runner.run("_modyn_client", str(EXAMPLE_PIPELINE), str(NO_FILE))
    assert not ret.success


@patch.object(Client, "__init__", noop_constructor_mock)
def test_supervisor_script_fails_on_non_existing_pipeline_config(script_runner):
    assert not NO_FILE.is_file(), "File that shouldn't exist exists."
    ret = script_runner.run("_modyn_client", str(NO_FILE), str(EXAMPLE_SYSTEM_CONFIG))
    assert not ret.success
