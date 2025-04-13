import os
import pathlib
import shutil
import tempfile

import pytest
from sqlalchemy.sql import Select

from modyn.config import PresamplingConfig
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.selector.internal.selector_strategies.presampling_strategies.original_set_strategy import (
    OriginalSetPresamplingStrategy,
)
from modyn.selector.internal.storage_backend.database.database_storage_backend import DatabaseStorageBackend

# Setup temporary paths for the database and trigger sample directory
database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"
TMP_DIR = tempfile.mkdtemp()


def get_config():
    # Create a configuration for the OriginalSet strategy. The ratio value is used by the inherited get_target_size.
    return PresamplingConfig(ratio=50, strategy="OriginalSet")


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


def test_constructor():
    strat = OriginalSetPresamplingStrategy(
        get_config(),
        get_minimal_modyn_config(),
        pipeline_id=10,
        storage_backend=DatabaseStorageBackend(0, get_minimal_modyn_config(), 123),
    )
    # The strategy should require a trigger dataset size.
    assert strat.requires_trigger_dataset_size


def test_get_presampling_query_trigger_zero():
    strat = OriginalSetPresamplingStrategy(
        get_config(),
        get_minimal_modyn_config(),
        pipeline_id=10,
        storage_backend=DatabaseStorageBackend(0, get_minimal_modyn_config(), 123),
    )
    # For the first trigger (next_trigger_id == 0) the query should simply select based on pipeline_id and order by timestamp.
    stmt: Select = strat.get_presampling_query(
        next_trigger_id=0, tail_triggers=None, limit=None, trigger_dataset_size=10
    )
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    # Check that the query orders by timestamp and does not include a union.
    assert "ORDER BY" in compiled.upper()
    assert "timestamp" in compiled
    assert "UNION" not in compiled.upper()


def test_get_presampling_query_trigger_non_zero():
    strat = OriginalSetPresamplingStrategy(
        get_config(),
        get_minimal_modyn_config(),
        pipeline_id=10,
        storage_backend=DatabaseStorageBackend(0, get_minimal_modyn_config(), 123),
    )
    # For non-zero next_trigger_id, the query should combine two subqueries (using UNION)
    stmt: Select = strat.get_presampling_query(
        next_trigger_id=5, tail_triggers=None, limit=None, trigger_dataset_size=20
    )
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    # Check that the union of subqueries is present and that the first subquery uses a LIMIT reflecting the target size.
    assert "UNION" in compiled.upper()
    # With a ratio of 50 and trigger_dataset_size of 20, the target size should be 10.
    assert "LIMIT 10" in compiled.upper() or "limit 10" in compiled


def test_get_presampling_query_with_limit():
    strat = OriginalSetPresamplingStrategy(
        get_config(),
        get_minimal_modyn_config(),
        pipeline_id=10,
        storage_backend=DatabaseStorageBackend(0, get_minimal_modyn_config(), 123),
    )
    # When a limit is provided, the target size should be the minimum of the calculated value and the limit.
    # For trigger_dataset_size 50 and ratio 50, the calculated target would be 25. With limit 15, target should be 15.
    stmt: Select = strat.get_presampling_query(next_trigger_id=3, tail_triggers=None, limit=15, trigger_dataset_size=50)
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "LIMIT 15" in compiled.upper() or "limit 15" in compiled


def test_get_presampling_query_invalid_trigger_dataset_size():
    strat = OriginalSetPresamplingStrategy(
        get_config(),
        get_minimal_modyn_config(),
        pipeline_id=10,
        storage_backend=DatabaseStorageBackend(0, get_minimal_modyn_config(), 123),
    )
    # Assert that passing a None or negative trigger_dataset_size raises an assertion error.
    with pytest.raises(AssertionError):
        strat.get_presampling_query(next_trigger_id=5, tail_triggers=None, limit=None, trigger_dataset_size=None)
    with pytest.raises(AssertionError):
        strat.get_presampling_query(next_trigger_id=5, tail_triggers=None, limit=None, trigger_dataset_size=-1)
