from unittest.mock import patch

import pytest
from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector
from modyn.trainer_server.internal.mocks.mock_metadata_processor import MockMetadataProcessorServer


def get_metadata_collector():
    return MetadataCollector(pipeline_id=0, trigger_id=1)


def test_add_per_sample_metadata_for_batch_fail():
    metadata_collector = get_metadata_collector()
    with pytest.raises(AssertionError):
        metadata_collector.add_per_sample_metadata_for_batch("mock_metric", ["0", "1", "2"], [0, 1, 2, 3, 4])


def test_add_per_sample_metadata_for_batch():
    metadata_collector = get_metadata_collector()
    metadata_collector.add_per_sample_metadata_for_batch("mock_metric_0", ["0", "1", "2"], [0.0, 1.0, 2.0])
    assert "mock_metric_0" in metadata_collector._per_sample_metadata_dict
    assert metadata_collector._per_sample_metadata_dict["mock_metric_0"] == {"0": 0.0, "1": 1.0, "2": 2.0}

    metadata_collector.add_per_sample_metadata_for_batch("mock_metric_0", ["3", "4", "5"], [3.0, 4.0, 5.0])
    assert metadata_collector._per_sample_metadata_dict["mock_metric_0"] == {
        "0": 0.0,
        "1": 1.0,
        "2": 2.0,
        "3": 3.0,
        "4": 4.0,
        "5": 5.0,
    }

    metadata_collector.add_per_sample_metadata_for_batch("mock_metric_1", ["1", "2", "3"], [1.0, 2.0, 3.0])
    assert "mock_metric_1" in metadata_collector._per_sample_metadata_dict
    assert metadata_collector._per_sample_metadata_dict["mock_metric_0"] == {
        "0": 0.0,
        "1": 1.0,
        "2": 2.0,
        "3": 3.0,
        "4": 4.0,
        "5": 5.0,
    }
    assert metadata_collector._per_sample_metadata_dict["mock_metric_1"] == {
        "1": 1.0,
        "2": 2.0,
        "3": 3.0,
    }


def test_add_per_trigger_metadata():
    metadata_collector = get_metadata_collector()
    metadata_collector.add_per_trigger_metadata("mock_metric_0", 0.0)
    metadata_collector.add_per_trigger_metadata("mock_metric_1", 1.0)

    assert metadata_collector._per_trigger_metadata == {
        "mock_metric_0": 0.0,
        "mock_metric_1": 1.0,
    }


def test_send_metadata_nometadata():
    metadata_collector = get_metadata_collector()
    metadata_collector.send_metadata()
    assert not bool(metadata_collector._per_sample_metadata_dict)
    assert not bool(metadata_collector._per_trigger_metadata)


@patch.object(MockMetadataProcessorServer, "send_metadata", return_value=None)
def test_send_metadata_per_sample_only(test_send_metadata):
    metadata_collector = get_metadata_collector()
    metadata_collector.add_per_sample_metadata_for_batch("loss", ["0", "1", "2"], [0.0, 1.0, 2.0])

    metadata_collector.send_metadata()
    test_send_metadata.assert_called_once()
    assert not bool(metadata_collector._per_sample_metadata_dict)
    assert not bool(metadata_collector._per_trigger_metadata)


@patch.object(MockMetadataProcessorServer, "send_metadata", return_value=None)
def test_send_metadata_per_trigger_only(test_send_metadata):
    metadata_collector = get_metadata_collector()
    metadata_collector.add_per_trigger_metadata("loss", 42.42)

    metadata_collector.send_metadata()
    test_send_metadata.assert_called_once()
    assert not bool(metadata_collector._per_sample_metadata_dict)
    assert not bool(metadata_collector._per_trigger_metadata)


@patch.object(MockMetadataProcessorServer, "send_metadata", return_value=None)
def test_send_metadata_all(test_send_metadata):
    metadata_collector = get_metadata_collector()
    metadata_collector.add_per_sample_metadata_for_batch("loss", ["0", "1", "2"], [0.0, 1.0, 2.0])
    metadata_collector.add_per_trigger_metadata("loss", 42.42)
    metadata_collector.send_metadata()
    assert test_send_metadata.call_count == 2
