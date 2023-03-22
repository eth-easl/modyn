# pylint: disable=no-value-for-parameter,redefined-outer-name,abstract-class-instantiated
import os
import pathlib
from math import isclose
from unittest.mock import MagicMock, patch

import pytest
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SampleTrainingMetadata, TriggerTrainingMetadata

# pylint: disable-next=no-name-in-module
from modyn.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (  # noqa: E402, E501
    PerSampleMetadata,
    PerTriggerMetadata,
)
from modyn.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"

TRIGGER_METADATA = PerTriggerMetadata(loss=0.05)
SAMPLE_METADATA = [PerSampleMetadata(sample_id="s1", loss=0.1)]


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


@patch.multiple(AbstractProcessorStrategy, __abstractmethods__=set())
def test_constructor():
    strat = AbstractProcessorStrategy(get_minimal_modyn_config(), 56)
    assert strat.pipeline_id == 56


@patch.multiple(AbstractProcessorStrategy, __abstractmethods__=set())
@patch.object(AbstractProcessorStrategy, "process_trigger_metadata")
@patch.object(AbstractProcessorStrategy, "process_sample_metadata")
@patch.object(AbstractProcessorStrategy, "persist_metadata")
def test_process_training_metadata(
    test__persist_metadata: MagicMock,
    test__process_sample_metadata: MagicMock,
    test__process_trigger_metadata: MagicMock,
):
    strat = AbstractProcessorStrategy(get_minimal_modyn_config(), 56)

    test__process_trigger_metadata.return_value = {"loss": 0.05}
    test__process_sample_metadata.return_value = [{"sample_id": "s1", "loss": 0.1}]

    strat.process_training_metadata(1, TRIGGER_METADATA, SAMPLE_METADATA)

    test__process_trigger_metadata.assert_called_once_with(TRIGGER_METADATA)
    test__process_sample_metadata.assert_called_once_with(SAMPLE_METADATA)
    test__persist_metadata.assert_called_once_with(1, {"loss": 0.05}, [{"sample_id": "s1", "loss": 0.1}])


@patch.multiple(AbstractProcessorStrategy, __abstractmethods__=set())
def test_process_trigger_metadata():
    strat = AbstractProcessorStrategy(get_minimal_modyn_config(), 56)
    with pytest.raises(NotImplementedError):
        strat.process_trigger_metadata(TRIGGER_METADATA)


@patch.multiple(AbstractProcessorStrategy, __abstractmethods__=set())
def test_process_sample_metadata():
    strat = AbstractProcessorStrategy(get_minimal_modyn_config(), 56)
    with pytest.raises(NotImplementedError):
        strat.process_sample_metadata(SAMPLE_METADATA)


@patch.multiple(AbstractProcessorStrategy, __abstractmethods__=set())
def test_persist_metadata():
    strat = AbstractProcessorStrategy(get_minimal_modyn_config(), 56)
    strat.persist_metadata(1, {"loss": 0.05}, [{"sample_id": "s1", "loss": 0.1}])

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        data = database.session.query(
            TriggerTrainingMetadata.trigger_id,
            TriggerTrainingMetadata.pipeline_id,
            TriggerTrainingMetadata.overall_loss,
        ).all()

        _, _, loss = zip(*data)
        assert len(loss) == 1, f"Expected 1 entry for trigger metadata, received {len(loss)}"
        assert loss[0] == 0.05, f"Expected overall loss 0.05, found {loss[0]}"

        data = database.session.query(
            SampleTrainingMetadata.pipeline_id,
            SampleTrainingMetadata.trigger_id,
            SampleTrainingMetadata.sample_key,
            SampleTrainingMetadata.loss,
        ).all()

        _, _, keys, loss = zip(*data)
        assert len(keys) == 1, f"Expected 1 entry for sample metadata, found {len(keys)}"
        assert keys[0] == "s1", f"Expected sample key s1, found {keys[0]}"
        assert isclose(loss[0], 0.1, rel_tol=1e-5), f"Expected sample loss 0.1, found {loss[0]}"
