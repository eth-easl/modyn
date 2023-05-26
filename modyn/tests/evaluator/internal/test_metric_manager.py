import pytest
from modyn.evaluator.internal.metric_manager import MetricManager


def test_metric_unavailability():
    manager = MetricManager()

    with pytest.raises(NotImplementedError):
        manager.get_evaluation_metric("UnknownMetric")


def test_metric_availability():
    manager = MetricManager()

    acc_metric = manager.get_evaluation_metric("AccuracyMetric")

    assert acc_metric
