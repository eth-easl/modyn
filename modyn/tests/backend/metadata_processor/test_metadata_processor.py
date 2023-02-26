# pylint: disable=no-value-for-parameter,redefined-outer-name
from unittest.mock import MagicMock, patch
from typing import Iterable, Optional

import pytest

from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (
    PerSampleMetadata,
    PerTriggerMetadata,
)
from modyn.backend.metadata_processor.metadata_processor import MetadataProcessor
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import (
    AbstractProcessorStrategy,
)

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
    processor = MetadataProcessor(MockStrategy(), 56)
    assert processor.pipeline_id == 56


@patch.object(MockStrategy, "process_training_metadata")
def test_process_training_metadata(test_process_training_metadata: MagicMock):
    processor = MetadataProcessor(MockStrategy(), 56)
    processor.process_training_metadata(4, PerTriggerMetadata(loss=0.05), [PerSampleMetadata(sample_id="s1", loss=0.1)])
    test_process_training_metadata.assert_called_once_with(4, PerTriggerMetadata(loss=0.05), [PerSampleMetadata(sample_id="s1", loss=0.1)])