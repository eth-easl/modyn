import os
import pathlib
import shutil
import tempfile

import pytest
from modyn.config import CoresetStrategyConfig, PresamplingConfig
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.selector.internal.selector_strategies import CoresetStrategy
from modyn.selector.internal.selector_strategies.presampling_strategies import RandomNoReplacementPresamplingStrategy

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"

TMP_DIR = tempfile.mkdtemp()


def get_config():
    return CoresetStrategyConfig(
        tail_triggers=None,
        presampling_config=PresamplingConfig(ratio=25, strategy="RandomNoReplacement"),
        limit=-1,
        maximum_keys_in_memory=1000,
    )


def get_minimal_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "hostname": "",
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


def insert_data(strat, base_index=0, size=200):
    strat.inform_data(
        range(base_index, base_index + size),
        range(base_index, base_index + size),
        [0, 1] * int((size / 2)),
    )


def test_sampling():
    modyn_config = get_minimal_modyn_config()
    strat = CoresetStrategy(get_config(), modyn_config, 0)

    strat.presampling_strategy: RandomNoReplacementPresamplingStrategy

    insert_data(strat, 0)

    presampler = strat.presampling_strategy

    samples_per_trigger = [None] * 6
    all_the_samples = []

    for i in range(4):
        query = presampler.get_presampling_query(i, None, None, 200)

        with MetadataDatabaseConnection(modyn_config) as database:
            result = database.session.execute(query).all()
            assert len(result) == 50
            selected_keys = [el[0] for el in result]
            samples_per_trigger[i] = selected_keys.copy()
            all_the_samples += selected_keys

    # check that every point has been sampled exactly once
    assert len(set(all_the_samples)) == len(all_the_samples) == 200
    # check that the last trigger
    assert presampler.last_complete_trigger == 0

    # now there are no samples. Let's add 20 but last_complete_trigger should be reset
    # so we expect to have 20 new samples and 30 samples from before

    insert_data(strat, 1000, size=20)
    query = presampler.get_presampling_query(5, None, None, 200)

    with MetadataDatabaseConnection(modyn_config) as database:
        result = database.session.execute(query).all()
        assert len(result) == 50
        selected_keys = [el[0] for el in result]
        new_samples = selected_keys

    assert len(set(new_samples).intersection(set(all_the_samples))) == 30
    assert len(set(new_samples).difference(set(all_the_samples))) == 20
