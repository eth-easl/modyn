import os
import pathlib
import shutil
import tempfile

import pytest
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.selector.internal.selector_strategies import RandomPresamplingStrategy

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


def test_init_random():
    # Test init works

    with pytest.raises(ValueError):
        RandomPresamplingStrategy(
            {"limit": -1, "reset_after_trigger": False, "presampling_ratio": 80, "downsampled_batch_ratio": 10},
            get_minimal_modyn_config(),
            42,
            1000,
        )

    strat = RandomPresamplingStrategy(
        {"limit": -1, "reset_after_trigger": False, "presampling_ratio": 80},
        get_minimal_modyn_config(),
        42,
        1000,
    )

    assert strat._pipeline_id == 42
    assert not hasattr(strat, "downsampled_batch_size")
