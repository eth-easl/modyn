# pylint: disable=no-value-for-parameter,redefined-outer-name,abstract-class-instantiated
import os
import pathlib
import shutil
import tempfile
from unittest.mock import patch

import pytest
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"

TMP_DIR = tempfile.mkdtemp()


def get_minimal_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "host": "",
            "port": "0",
            "database": f"{database_path}",
        },
        "selector": {"insertion_threads": 8, "trigger_sample_directory": TMP_DIR},
    }


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield

    os.remove(database_path)
    shutil.rmtree(TMP_DIR)


@patch.multiple(AbstractDownsamplingStrategy, __abstractmethods__=set())
def test_constructor_invalid_config():
    # missing downsampling_ratio
    downsampling_config = {
        "sample_then_batch": True,
    }

    with pytest.raises(ValueError):
        AbstractDownsamplingStrategy(downsampling_config, 1000)

    # float downsampling_ratio
    downsampling_config = {"sample_then_batch": True, "ratio": 0.18}
    with pytest.raises(ValueError):
        AbstractDownsamplingStrategy(downsampling_config, 1000)

    downsampling_config = {"sample_then_batch": True, "ratio": 10}

    ads = AbstractDownsamplingStrategy(downsampling_config, 1000)

    assert ads.requires_remote_computation
    assert ads.downsampling_ratio == 10
