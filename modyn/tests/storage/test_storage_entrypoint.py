"""
This tests that the entry point script for the storage
successfully runs through. This is _not_ the place for an integration test.
"""
import os
import pathlib
from unittest.mock import patch
from modyn.storage import Storage

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

EXAMPLE_SYSTEM_CONFIG = SCRIPT_PATH.parent.parent.parent / "config" / "examples" / "modyn_config.yaml"

NO_FILE = SCRIPT_PATH.parent / "thisshouldnot.exist"


def noop_constructor_mock(self, modyn_config: dict) -> None:  # pylint: disable=unused-argument
    pass


def noop_run(self) -> None:  # pylint: disable=unused-argument
    pass


@patch.object(Storage, '__init__', noop_constructor_mock)
@patch.object(Storage, 'run', noop_run)
def test_storage_script_runs(script_runner):
    ret = script_runner.run('_modyn_storage', str(EXAMPLE_SYSTEM_CONFIG))
    assert ret.success


@patch.object(Storage, '__init__', noop_constructor_mock)
def test_storage_script_fails_on_non_existing_system_config(script_runner):
    assert not NO_FILE.is_file(), "File that shouldn't exist exists."
    ret = script_runner.run('_modyn_storage', str(NO_FILE))
    assert not ret.success
