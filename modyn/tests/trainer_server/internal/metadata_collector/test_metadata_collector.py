from unittest.mock import patch

import pytest

from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector
from modyn.trainer_server.internal.mocks.mock_metadata_processor import MockMetadataProcessorServer
from modyn.trainer_server.internal.utils.metric_type import MetricType


def get_metadata_collector():
    return MetadataCollector(pipeline_id=0, trigger_id=1)


def test_add_per_sample_metadata_for_batch_fail():
    metadata_collector = get_metadata_collector()
    with pytest.raises(AssertionError):
        metadata_collector.add_per_sample_metadata_for_batch(MetricType.LOSS, ["0", "1", "2"], [0, 1, 2, 3, 4])


def test_add_per_sample_metadata_for_batch():
    metadata_collector = get_metadata_collector()
    metadata_collector.add_per_sample_metadata_for_batch(MetricType.LOSS, ["0", "1", "2"], [0.0, 1.0, 2.0])
    assert MetricType.LOSS in metadata_collector._per_sample_metadata_dict
    assert metadata_collector._per_sample_metadata_dict[MetricType.LOSS] == {"0": 0.0, "1": 1.0, "2": 2.0}

    metadata_collector.add_per_sample_metadata_for_batch(MetricType.LOSS, ["3", "4", "5"], [3.0, 4.0, 5.0])
    assert metadata_collector._per_sample_metadata_dict[MetricType.LOSS] == {
        "0": 0.0,
        "1": 1.0,
        "2": 2.0,
        "3": 3.0,
        "4": 4.0,
        "5": 5.0,
    }


def test_add_per_trigger_metadata():
    metadata_collector = get_metadata_collector()
    metadata_collector.add_per_trigger_metadata(MetricType.LOSS, 0.0)

    assert metadata_collector._per_trigger_metadata == {
        MetricType.LOSS: 0.0,
    }


@patch.object(MetadataCollector, "send_loss")
def test_send_metadata(send_loss_mock):
    metadata_collector = get_metadata_collector()
    metadata_collector.send_metadata(MetricType.LOSS)
    send_loss_mock.assert_called_once()


@patch.object(MockMetadataProcessorServer, "send_metadata")
def test_send_loss(send_metadata_mock):
    metadata_collector = get_metadata_collector()
    metadata_collector._per_sample_metadata_dict[MetricType.LOSS] = {"0": 0.0, "1": 1.0, "2": 2.0}
    metadata_collector._per_trigger_metadata[MetricType.LOSS] = 0.1
    metadata_collector.send_loss()
    send_metadata_mock.assert_called_once()


def test_cleanup():
    metadata_collector = get_metadata_collector()
    metadata_collector._per_sample_metadata_dict[MetricType.LOSS] = {"0": 0.0, "1": 1.0, "2": 2.0}
    metadata_collector._per_trigger_metadata[MetricType.LOSS] = 0.1
    metadata_collector.cleanup()
    assert not metadata_collector._per_sample_metadata_dict
    assert not metadata_collector._per_trigger_metadata
