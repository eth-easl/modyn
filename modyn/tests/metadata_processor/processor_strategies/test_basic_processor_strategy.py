# pylint: disable=no-value-for-parameter,redefined-outer-name,abstract-class-instantiated
import os
import pathlib
from math import isclose

import pytest
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SampleTrainingMetadata, TriggerTrainingMetadata

# pylint: disable-next=no-name-in-module
from modyn.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (  # noqa: E402, E501
    PerSampleMetadata,
    PerTriggerMetadata,
)
from modyn.metadata_processor.processor_strategies.basic_processor_strategy import BasicProcessorStrategy

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"

TRIGGER_METADATA = PerTriggerMetadata(loss=0.05)
SAMPLE_METADATA = [PerSampleMetadata(sample_id="s1", loss=0.1)]


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
    }


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield

    os.remove(database_path)


TRIGGER_METADATA = PerTriggerMetadata(loss=0.05)
SAMPLE_METADATA = [PerSampleMetadata(sample_id="s1", loss=0.1), PerSampleMetadata(sample_id="s2", loss=0.2)]


def test_constructor():
    BasicProcessorStrategy(get_minimal_modyn_config(), 56)


def test_process_trigger_metadata():
    strat = BasicProcessorStrategy(get_minimal_modyn_config(), 56)
    assert strat.process_trigger_metadata(TRIGGER_METADATA) == {"loss": pytest.approx(0.05)}


def test_process_sample_metadata():
    strat = BasicProcessorStrategy(get_minimal_modyn_config(), 56)
    assert strat.process_sample_metadata(SAMPLE_METADATA) == [
        {"sample_id": "s1", "loss": pytest.approx(0.1)},
        {"sample_id": "s2", "loss": pytest.approx(0.2)},
    ]


def test_process_training_metadata():
    strat = BasicProcessorStrategy(get_minimal_modyn_config(), 56)
    strat.process_training_metadata(1, TRIGGER_METADATA, SAMPLE_METADATA)

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(
            TriggerTrainingMetadata.trigger_id,
            TriggerTrainingMetadata.pipeline_id,
            TriggerTrainingMetadata.overall_loss,
        ).all()

        _, _, loss = zip(*data)
        assert len(loss) == 1, f"Expected 1 entry for trigger metadata, received {len(loss)}"
        assert loss[0] == TRIGGER_METADATA.loss, f"Expected overall loss {TRIGGER_METADATA.loss}, found {loss[0]}"

        data = database.session.query(
            SampleTrainingMetadata.pipeline_id,
            SampleTrainingMetadata.trigger_id,
            SampleTrainingMetadata.sample_key,
            SampleTrainingMetadata.loss,
        ).all()

        _, _, keys, loss = zip(*data)
        assert len(keys) == 2, f"Expected 2 entries for sample metadata, found {len(keys)}"
        assert keys == ("s1", "s2"), f"Expected sample keys [s1, s2], found {keys}"
        assert isclose(loss[0], 0.1, rel_tol=1e-5), f"Expected sample loss 0.1, found {loss[0]}"
        assert isclose(loss[1], 0.2, rel_tol=1e-5), f"Expected sample loss 0.2, found {loss[1]}"
