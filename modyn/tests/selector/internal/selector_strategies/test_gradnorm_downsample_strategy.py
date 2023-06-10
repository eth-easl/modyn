import os
import pathlib
import shutil
import tempfile

import pytest
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.selector.internal.selector_strategies import GradNormDownsamplingStrategy

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


def test_init_gradnorm():
    # Test init works
    strat = GradNormDownsamplingStrategy(
        {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_ratio": 80,
            "downsampled_batch_ratio": 50,
            "sample_then_batch": False,
        },
        get_minimal_modyn_config(),
        42,
        1000,
    )

    assert strat.downsampled_batch_ratio == 50
    assert strat._pipeline_id == 42
    assert isinstance(strat.get_downsampling_strategy(), str)


def test_command_gradnorm():
    # Test init works
    strat = GradNormDownsamplingStrategy(
        {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_ratio": 80,
            "downsampled_batch_ratio": 10,
            "sample_then_batch": False,
        },
        get_minimal_modyn_config(),
        42,
        1000,
    )

    name = strat.get_downsampling_strategy()
    params = strat.get_downsampling_params()
    assert isinstance(name, str)
    assert name == "RemoteGradNormDownsampling"
    assert "downsampled_batch_ratio" in params
    assert params["downsampled_batch_ratio"] == 10
