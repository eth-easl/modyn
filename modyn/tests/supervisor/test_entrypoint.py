# pylint: disable=unused-argument
""" This tests that the entry point script for the supervisor
successfully runs through. This is _not_ the place for an integration test.
"""
import os
import pathlib
import tempfile
import typing
from unittest.mock import patch

from modyn.supervisor import Supervisor

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

EXAMPLE_PIPELINE = SCRIPT_PATH.parent.parent.parent / "config" / "examples" / "example-pipeline.yaml"
EXAMPLE_SYSTEM_CONFIG = SCRIPT_PATH.parent.parent.parent / "config" / "examples" / "modyn_config.yaml"

NO_FILE = SCRIPT_PATH.parent / "thisshouldnot.exist"


def noop_constructor_mock(
    self,
    pipeline_config: dict,
    modyn_config: dict,
    eval_directory: pathlib.Path,
    start_replay_at: typing.Optional[int] = None,
    stop_replay_at: typing.Optional[int] = None,
    maximum_triggers: typing.Optional[int] = None,
    valuation_matrix: bool = False,
) -> None:
    pass


def noop_pipeline(self) -> None:
    pass


@patch.object(Supervisor, "__init__", noop_constructor_mock)
@patch.object(Supervisor, "pipeline", noop_pipeline)
def test_supervisor_script_runs(script_runner):
    with tempfile.TemporaryDirectory() as eval_dir:
        ret = script_runner.run("_modyn_supervisor", str(EXAMPLE_PIPELINE), str(EXAMPLE_SYSTEM_CONFIG), str(eval_dir))
        assert ret.success


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_supervisor_script_fails_on_non_existing_system_config(script_runner):
    with tempfile.TemporaryDirectory() as eval_dir:
        assert not NO_FILE.is_file(), "File that shouldn't exist exists."
        ret = script_runner.run("_modyn_supervisor", str(EXAMPLE_PIPELINE), str(NO_FILE), str(eval_dir))
        assert not ret.success


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_supervisor_script_fails_on_non_existing_pipeline_config(script_runner):
    with tempfile.TemporaryDirectory() as eval_dir:
        assert not NO_FILE.is_file(), "File that shouldn't exist exists."
        ret = script_runner.run("_modyn_supervisor", str(NO_FILE), str(EXAMPLE_SYSTEM_CONFIG), str(eval_dir))
        assert not ret.success


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_supervisor_script_fails_on_non_existing_eval_dir(script_runner):
    eval_dir = pathlib.Path("unknownfolder")
    assert not eval_dir.is_dir(), "Directory that shouldn't exist exists."
    ret = script_runner.run("_modyn_supervisor", str(NO_FILE), str(EXAMPLE_SYSTEM_CONFIG), str(eval_dir))
    assert not ret.success
