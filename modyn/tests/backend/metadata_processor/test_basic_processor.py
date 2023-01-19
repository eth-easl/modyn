# pylint: disable=no-value-for-parameter
from unittest.mock import patch

import pytest
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy
from modyn.backend.metadata_processor.processor_strategies.basic_processor_strategy import BasicProcessorStrategy

TEST_TRAINING_ID = 10
TEST_DATA = """
{
    "key1": "value1",
    "key2": "value2",
    "key3": "value3"
}
"""
TEST_NONJSON_DATA = """key1: value1"""


class MockGRPCHandler:
    def __init__(self, modyn_config: dict) -> None:
        self.config = modyn_config
        self.connected_to_database = True

        self.training_id = []
        self.data = []

    def set_metadata(self, training_id: int, data: dict) -> None:
        self.training_id.append(training_id)
        self.data.append(data)


def grpchandler_constructor_mock(self, modyn_config: dict) -> None:
    self.grpc = MockGRPCHandler(modyn_config)


@patch.object(AbstractProcessorStrategy, "__init__", grpchandler_constructor_mock)
def test_process_post_training_metadata():
    strategy = BasicProcessorStrategy(None)
    strategy.process_post_training_metadata(TEST_TRAINING_ID, TEST_DATA)

    assert len(strategy.grpc.training_id) == 1
    assert len(strategy.grpc.data) == 1
    assert strategy.grpc.training_id[0] == 10
    assert strategy.grpc.data[0] == {
        "keys": ["key1", "key2", "key3"],
        "seen": [True, True, True],
        "data": ["value1", "value2", "value3"],
    }


@patch.object(AbstractProcessorStrategy, "__init__", grpchandler_constructor_mock)
def test_strategy_throws_on_nonjson_data():
    strategy = BasicProcessorStrategy(None)
    with pytest.raises(ValueError):
        strategy.process_post_training_metadata(TEST_TRAINING_ID, TEST_NONJSON_DATA)
