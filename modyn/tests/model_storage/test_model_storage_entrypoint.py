# pylint: disable=unused-argument
"""This tests that the entry point script for the model storage successfully
runs through.

This is _not_ the place for an integration test.
"""

import os
import pathlib
from unittest.mock import patch

from modyn.model_storage import ModelStorage

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

EXAMPLE_SYSTEM_CONFIG = SCRIPT_PATH.parent.parent.parent / "config" / "examples" / "modyn_config.yaml"

NO_FILE = SCRIPT_PATH.parent / "unknownfile.txt"


def noop_constructor_mock(self, config: dict) -> None:  # pylint: disable=unused-argument
    pass


def noop_run(self) -> None:  # pylint: disable=unused-argument
    pass


@patch.object(ModelStorage, "__init__", noop_constructor_mock)
@patch.object(ModelStorage, "run", noop_run)
def test_model_storage_script_runs(script_runner):
    ret = script_runner.run("_modyn_model_storage", str(EXAMPLE_SYSTEM_CONFIG))
    assert ret.success


@patch.object(ModelStorage, "__init__", noop_constructor_mock)
def test_model_storage_script_fails_on_non_existing_system_config(script_runner):
    assert not NO_FILE.is_file(), "File that shouldn't exist exists."
    ret = script_runner.run("_modyn_model_storage", str(NO_FILE))
    assert not ret.success
