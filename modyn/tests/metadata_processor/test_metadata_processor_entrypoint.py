# pylint: disable=unused-argument
"""This tests that the entrypoint script for the metadata processor
successfully runs through."""

import os
import pathlib
from unittest.mock import patch

from modyn.metadata_processor.internal.grpc.metadata_processor_server import MetadataProcessorServer

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

EXAMPLE_SYSTEM_CONFIG = SCRIPT_PATH.parent.parent.parent / "config" / "examples" / "modyn_config.yaml"

NO_FILE = SCRIPT_PATH.parent / "thisshouldnot.exist"


def noop_constructor_mock(self, modyn_config: dict) -> None:
    pass


def noop_run(self) -> None:
    pass


@patch.object(MetadataProcessorServer, "__init__", noop_constructor_mock)
@patch.object(MetadataProcessorServer, "run", noop_run)
def test_processor_script_runs(script_runner):
    ret = script_runner.run("_modyn_metadata_processor", str(EXAMPLE_SYSTEM_CONFIG))
    assert ret.success


@patch.object(MetadataProcessorServer, "__init__", noop_constructor_mock)
def test_processor_script_fails_on_non_existing_system_config(script_runner):
    assert not NO_FILE.is_file(), "File that shouldn't exist exists."
    ret = script_runner.run("_modyn_metadata_processor", str(NO_FILE))
    assert not ret.success
