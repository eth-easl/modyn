import os
import pathlib
import shutil
import tempfile

import pytest
from sqlalchemy import select

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.storage_backend.database import DatabaseStorageBackend
from modyn.utils.utils import flatten

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"
TMP_DIR = tempfile.mkdtemp()


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


def test_persist_samples():
    backend = DatabaseStorageBackend(42, get_minimal_modyn_config(), 1000)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(
            SelectorStateMetadata.sample_key,
            SelectorStateMetadata.timestamp,
            SelectorStateMetadata.label,
            SelectorStateMetadata.pipeline_id,
            SelectorStateMetadata.used,
        ).all()

        assert len(data) == 3

        keys, timestamps, labels, pipeline_ids, useds = zip(*data)

        assert not any(useds)
        for pip_id in pipeline_ids:
            assert pip_id == 42

        assert keys[0] == 10 and keys[1] == 11 and keys[2] == 12
        assert timestamps[0] == 0 and timestamps[1] == 1 and timestamps[2] == 2
        assert labels[0] == 40 and labels[1] == 41 and labels[2] == 42


def test_get_data_since_trigger():
    backend = DatabaseStorageBackend(42, get_minimal_modyn_config(), 2)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])
    backend.persist_samples(1, [13, 14, 15], [3, 4, 5], [40, 41, 42])
    backend.persist_samples(2, [16, 17, 18], [6, 7, 8], [40, 41, 42])

    data_since_trigger_one = [keys for keys, _ in backend.get_data_since_trigger(1)]
    assert len(data_since_trigger_one) == 3  # Validate partitioning
    assert flatten(data_since_trigger_one) == [
        13,
        14,
        15,
        16,
        17,
        18,
    ]  # Validate content


def test_get_trigger_data():
    backend = DatabaseStorageBackend(42, get_minimal_modyn_config(), 2)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])
    backend.persist_samples(1, [13, 14, 15], [3, 4, 5], [40, 41, 42])
    backend.persist_samples(2, [16, 17, 18], [6, 7, 8], [40, 41, 42])

    trigger_zero_data = [keys for keys, _ in backend.get_trigger_data(0)]
    trigger_one_data = [keys for keys, _ in backend.get_trigger_data(1)]
    trigger_three_data = [keys for keys, _ in backend.get_trigger_data(3)]

    assert len(trigger_zero_data) == 2  # Validate partitioning
    assert flatten(trigger_zero_data) == [10, 11, 12]  # Validate content

    assert len(trigger_one_data) == 2  # Validate partitioning
    assert flatten(trigger_one_data) == [13, 14, 15]  # Validate content

    assert len(trigger_three_data) == 0


def test_get_all_data():
    backend = DatabaseStorageBackend(42, get_minimal_modyn_config(), 2)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])
    backend.persist_samples(1, [13, 14, 15], [3, 4, 5], [40, 41, 42])
    backend.persist_samples(2, [16, 17, 18], [6, 7, 8], [40, 41, 42])

    all_data = [keys for keys, _ in backend.get_all_data()]

    assert len(all_data) == 5  # Validate partitioning
    assert flatten(all_data) == [10, 11, 12, 13, 14, 15, 16, 17, 18]  # Validate content


def test_get_available_labels_with_tail():
    backend = DatabaseStorageBackend(1, get_minimal_modyn_config(), 42)
    # Next trigger is 0
    assert sorted(backend.get_available_labels(0)) == []

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        # first trigger
        database.session.add(
            SelectorStateMetadata(pipeline_id=1, sample_key=0, seen_in_trigger_id=0, timestamp=0, label=1)
        )
        database.session.add(
            SelectorStateMetadata(pipeline_id=1, sample_key=1, seen_in_trigger_id=0, timestamp=0, label=18)
        )
        database.session.add(
            SelectorStateMetadata(pipeline_id=1, sample_key=2, seen_in_trigger_id=0, timestamp=0, label=1)
        )
        database.session.add(
            SelectorStateMetadata(pipeline_id=1, sample_key=3, seen_in_trigger_id=0, timestamp=0, label=0)
        )
        database.session.commit()

    # Next trigger is 1, trigger 0 has data now
    assert sorted(backend.get_available_labels(1)) == [0, 1, 18]

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        # second trigger
        database.session.add(
            SelectorStateMetadata(pipeline_id=1, sample_key=4, seen_in_trigger_id=1, timestamp=0, label=0)
        )
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1,
                sample_key=5,
                seen_in_trigger_id=1,
                timestamp=0,
                label=890,
            )
        )
        database.session.commit()

    # Next trigger is 2, trigger 1 has data now
    assert sorted(backend.get_available_labels(2, tail_triggers=0)) == [0, 890]


def test_get_available_labels_no_tail():
    backend = DatabaseStorageBackend(1, get_minimal_modyn_config(), 42)
    # Next trigger is 0
    assert sorted(backend.get_available_labels(0)) == []

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        # first trigger
        database.session.add(
            SelectorStateMetadata(pipeline_id=1, sample_key=0, seen_in_trigger_id=0, timestamp=0, label=1)
        )
        database.session.add(
            SelectorStateMetadata(pipeline_id=1, sample_key=1, seen_in_trigger_id=0, timestamp=0, label=18)
        )
        database.session.add(
            SelectorStateMetadata(pipeline_id=1, sample_key=2, seen_in_trigger_id=0, timestamp=0, label=1)
        )
        database.session.add(
            SelectorStateMetadata(pipeline_id=1, sample_key=3, seen_in_trigger_id=0, timestamp=0, label=0)
        )
        database.session.commit()

    # Next trigger is 1, trigger 0 has data now
    assert sorted(backend.get_available_labels(1)) == [0, 1, 18]

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        # second trigger
        database.session.add(
            SelectorStateMetadata(pipeline_id=1, sample_key=4, seen_in_trigger_id=1, timestamp=0, label=0)
        )
        database.session.add(
            SelectorStateMetadata(
                pipeline_id=1,
                sample_key=5,
                seen_in_trigger_id=1,
                timestamp=0,
                label=890,
            )
        )
        database.session.commit()

    # Next trigger is 2, trigger 1 has data now
    assert sorted(backend.get_available_labels(2)) == [0, 1, 18, 890]


# BEGIN TESTS OF INTERNAL FUNCTIONS #


def test__get_pipeline_data_yield_per():
    backend = DatabaseStorageBackend(42, get_minimal_modyn_config(), 2)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])

    all_data_yp1 = [keys for keys, _ in backend._get_pipeline_data((), yield_per=1)]
    all_data_yp2 = [keys for keys, _ in backend._get_pipeline_data((), yield_per=2)]
    all_data_yp3 = [keys for keys, _ in backend._get_pipeline_data((), yield_per=3)]

    assert flatten(all_data_yp1) == flatten(all_data_yp2)
    assert flatten(all_data_yp2) == flatten(all_data_yp3)
    assert flatten(all_data_yp3) == [10, 11, 12]

    assert len(all_data_yp1) == 3
    assert len(all_data_yp2) == 2
    assert len(all_data_yp3) == 1


def test__get_pipeline_data_filter():
    backend = DatabaseStorageBackend(42, get_minimal_modyn_config(), 2)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])

    filtered_data = [keys for keys, _ in backend._get_pipeline_data((SelectorStateMetadata.sample_key == 11,))]
    assert flatten(filtered_data) == [11]


# This is a test, we do not want to have this global at the module level
# As we delete it afterwards, this is safe
# pylint: disable=global-variable-undefined


def test__partitioned_execute_stmt():
    backend = DatabaseStorageBackend(42, get_minimal_modyn_config(), 2)
    backend.persist_samples(0, [10, 11, 12], [0, 1, 2], [40, 41, 42])

    # We need a global here such that it is available in the callback
    global dbsb_test_part_exec_callback_call_count
    dbsb_test_part_exec_callback_call_count = 0

    def callback(_):
        global dbsb_test_part_exec_callback_call_count
        dbsb_test_part_exec_callback_call_count += 1

    stmt = (
        select(SelectorStateMetadata.sample_key)
        .execution_options(yield_per=1)
        .filter(SelectorStateMetadata.pipeline_id == 42)
    )

    selected_data = [keys for keys, _ in backend._partitioned_execute_stmt(stmt, 1, callback)]
    assert flatten(selected_data) == [10, 11, 12]
    assert len(selected_data) == 3
    assert dbsb_test_part_exec_callback_call_count == 3

    del dbsb_test_part_exec_callback_call_count  # Cleanup global


class MockStorageStub:
    def __init__(self, channel) -> None:
        pass

    def Get(self, request):  # pylint: disable=invalid-name  # noqa: N802
        # Return a fake response with predetermined values.
        class FakeResponse:
            def __init__(self):
                self.keys = [10, 20]
                self.samples = [b"sample10", b"sample20"]
                self.labels = [100, 200]
                self.target = [b"target10", b"target20"]

        yield FakeResponse()


# --- Fixture: Create an instance of DatabaseStorageBackend ---
@pytest.fixture
def db_storage_backend():
    backend = DatabaseStorageBackend(42, get_minimal_modyn_config(), 2)
    return backend


# --- Test for _get_data_from_storage ---
def test_get_data_from_storage(db_storage_backend):
    """
    Verifies that _get_data_from_storage returns a 5-tuple in the proper order:
      (keys, samples, targets, labels, response_time)
    """
    # Patch _init_grpc so that no real connection is attempted.
    db_storage_backend._init_grpc = lambda worker_id=None: None
    # Override _storagestub with our fake stub.
    db_storage_backend._storagestub = MockStorageStub()
    results = list(db_storage_backend._get_data_from_storage(selector_keys=[10, 20], dataset_id="test_dataset"))

    # We expect exactly one response from our fake stub.
    assert len(results) == 1
    keys, samples, targets, labels, resp_time = results[0]
    assert keys == [10, 20]
    assert samples == [b"sample10", b"sample20"]
    assert targets == [b"target10", b"target20"]
    assert labels == [100, 200]
