import os
import pathlib
import shutil
import tempfile

import pytest
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.abstract_downsample_strategy import AbstractDownsampleStrategy
from modyn.selector.internal.selector_strategies.abstract_presample_strategy import AbstractPresampleStrategy

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


def get_config():
    return {"reset_after_trigger": False, "presampling_ratio": 50, "limit": -1, "downsampled_batch_size": 10, "sample_before_batch": False}


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield

    os.remove(database_path)
    shutil.rmtree(TMP_DIR)



def get_config_tail():
    return {
        "reset_after_trigger": False,
        "presampling_ratio": 50,
        "limit": -1,
        "downsampled_batch_size": 10,
        "tail_triggers": 1,
        "sample_before_batch": False
    }


def test_constructor():
    strat = AbstractDownsampleStrategy(get_config(), get_minimal_modyn_config(), 0, 1000)
    assert strat.presampling_ratio >= 0


def test_constructor_throws_on_invalid_config():
    conf = {"reset_after_trigger": False, "presampling_ratio": 50, "limit": -1, "sample_before_batch": False}

    with pytest.raises(ValueError):
        AbstractDownsampleStrategy(conf, get_minimal_modyn_config(), 0, 1000)

    conf = {"reset_after_trigger": False, "presampling_ratio": 50, "limit": -1, "downsampled_batch_size": 0.10, "sample_before_batch": False}

    with pytest.raises(ValueError):
        AbstractDownsampleStrategy(conf, get_minimal_modyn_config(), 0, 1000)

    conf = {"reset_after_trigger": False, "presampling_ratio": 50, "limit": -1, "downsampled_batch_size": 10, "sample_before_batch": False}
    ads = AbstractDownsampleStrategy(conf, get_minimal_modyn_config(), 0, 1000)

    assert ads._requires_remote_computation
    assert isinstance(ads, AbstractPresampleStrategy)
    assert isinstance(ads, AbstractSelectionStrategy)
