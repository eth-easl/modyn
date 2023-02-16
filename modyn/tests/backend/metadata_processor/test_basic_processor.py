# pylint: disable=no-value-for-parameter
import os
import pathlib
from math import isclose

import pytest
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SampleTrainingMetadata, TriggerTrainingMetadata

# pylint: disable-next=no-name-in-module
from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (  # noqa: E402, E501
    PerSampleMetadata,
    PerTriggerMetadata,
)
from modyn.backend.metadata_processor.processor_strategies.basic_processor_strategy import BasicProcessorStrategy

PIPELINE_ID = 1
TRIGGER_ID = 1
TRIGGER_METADATA = PerTriggerMetadata(loss=0.05)
SAMPLE_METADATA = [PerSampleMetadata(sample_id="s1", loss=0.1), PerSampleMetadata(sample_id="s2", loss=0.2)]

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"


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
    }


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield
    os.remove(database_path)


def test_constructor():
    strat = BasicProcessorStrategy(get_minimal_modyn_config())
    assert strat


def test_process_training_metadata():
    strat = BasicProcessorStrategy(get_minimal_modyn_config())
    strat.process_training_metadata(PIPELINE_ID, TRIGGER_ID, TRIGGER_METADATA, SAMPLE_METADATA)

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = (
            database.session.query(
                TriggerTrainingMetadata.trigger_id,
                TriggerTrainingMetadata.pipeline_id,
                TriggerTrainingMetadata.overall_loss,
            )
            .filter(TriggerTrainingMetadata.trigger_id == TRIGGER_ID)
            .filter(TriggerTrainingMetadata.pipeline_id == PIPELINE_ID)
            .all()
        )

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
