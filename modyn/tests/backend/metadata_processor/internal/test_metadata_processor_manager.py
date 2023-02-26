# pylint: disable=no-value-for-parameter,redefined-outer-name
from unittest.mock import MagicMock, patch
from typing import Iterable, Optional

import pytest

from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (
    PerSampleMetadata,
    PerTriggerMetadata,
)
from modyn.backend.metadata_processor.internal.metadata_processor_manager import MetadataProcessorManager
from modyn.backend.metadata_processor.metadata_processor import MetadataProcessor
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import (
    AbstractProcessorStrategy,
)


TRIGGER_METATDATA = PerTriggerMetadata(loss=0.05)
SAMPLE_METADATA = [PerSampleMetadata(sample_id="s1", loss=0.1)]


def get_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "user",
            "password": "pw",
            "database": "db",
            "host": "derhorst",
            "port": "1337",
        }
    }


class MockStrategy(AbstractProcessorStrategy):
    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def process_training_metadata(
        trigger_id: int,
        trigger_metadata: PerTriggerMetadata,
        sample_metadata: PerSampleMetadata
    ) -> None:
        pass

    def process_trigger_metadata(self, trigger_metadata: PerTriggerMetadata) -> Optional[dict]:
        pass

    def process_sample_metadata(self, sample_metadata: Iterable[PerSampleMetadata]) -> Optional[list[dict]]:
        pass


def test_constructor():
    manager = MetadataProcessorManager(get_modyn_config())


@patch.object(MetadataProcessorManager, "_instantiate_strategy")
def test_register_pipeline(test__instantiate_strategy: MagicMock):
    manager = MetadataProcessorManager(get_modyn_config())
    test__instantiate_strategy.return_value = MockStrategy()

    assert len(manager.processors) == 0

    manager.register_pipeline(56, "..")
    assert len(manager.processors) == 1

    assert 56 in manager.processors.keys()
    assert isinstance(manager.processors[56].strategy, MockStrategy)


@patch.object(MetadataProcessorManager, "_instantiate_strategy")
@patch.object(MetadataProcessor, "process_training_metadata")
def test_processor_training_metadata(
    metadata_processor_process_training_metadata: MagicMock, test__instantiate_strategy: MagicMock
):
    manager = MetadataProcessorManager(get_modyn_config())
    test__instantiate_strategy.return_value = MockStrategy()

    manager.register_pipeline(56, "..")

    with pytest.raises(ValueError):
        manager.process_training_metadata(60, 0, TRIGGER_METATDATA, SAMPLE_METADATA)

    manager.process_training_metadata(56, 1, TRIGGER_METATDATA, SAMPLE_METADATA)
    metadata_processor_process_training_metadata.assert_called_once_with(1, TRIGGER_METATDATA, SAMPLE_METADATA)