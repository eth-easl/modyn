# pylint: disable=no-value-for-parameter
import os
import pathlib
from math import isclose

import pytest

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


def test_constructor():
    strat = BasicProcessorStrategy()
    assert strat


def test_process_trigger_metadata():
    strat = BasicProcessorStrategy()
    strat.process_trigger_metadata(TRIGGER_METADATA)

def test_process_sample_metadata():
    strat = BasicProcessorStrategy()
    strat.process_sample_metadata(SAMPLE_METADATA)