# pylint: disable=unused-argument
""" This tests that the entry point script for the supervisor
successfully runs through. This is _not_ the place for an integration test.
"""
import os
import pathlib
import typing
from unittest.mock import patch
from modyn.backend.supervisor import Supervisor

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

EXAMPLE_PIPELINE = SCRIPT_PATH.parent.parent.parent.parent / "config" / "examples" / "example-pipeline.yaml"
EXAMPLE_SYSTEM_CONFIG = SCRIPT_PATH.parent.parent.parent.parent / "config" / "examples" / "modyn_config.yaml"

NO_FILE = SCRIPT_PATH.parent / "thisshouldnot.exist"


def noop_constructor_mock(self, pipeline_config: dict, modyn_config: dict, replay_at: typing.Optional[int]) -> None:
    pass


def noop_pipeline(self) -> None:
    pass


@patch.object(Supervisor, "__init__", noop_constructor_mock)
@patch.object(Supervisor, "pipeline", noop_pipeline)
def test_supervisor_script_runs(script_runner):
    ret = script_runner.run("_modyn_supervisor", str(EXAMPLE_PIPELINE), str(EXAMPLE_SYSTEM_CONFIG))
    assert ret.success


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_supervisor_script_fails_on_non_existing_system_config(script_runner):
    assert not NO_FILE.is_file(), "File that shouldn't exist exists."
    ret = script_runner.run("_modyn_supervisor", str(EXAMPLE_PIPELINE), str(NO_FILE))
    assert not ret.success


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_supervisor_script_fails_on_non_existing_pipeline_config(script_runner):
    assert not NO_FILE.is_file(), "File that shouldn't exist exists."
    ret = script_runner.run("_modyn_supervisor", str(NO_FILE), str(EXAMPLE_SYSTEM_CONFIG))
    assert not ret.success
