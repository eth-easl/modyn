import pytest
import torch
from modyn.evaluator.internal.metric_manager import MetricManager


def test_accuracy():
    manager = MetricManager()
    acc_metric = manager.get_evaluation_metric("AccuracyMetric")

    zeroes = torch.zeros(10)
    y_pred = torch.zeros(10)

    assert acc_metric.evaluate(zeroes, y_pred) == 1

    y_pred = torch.ones((1, 10))
    y_pred.squeeze_()

    assert acc_metric.evaluate(zeroes, y_pred) == 0

    y_pred[4:6] = 0

    assert acc_metric.evaluate(zeroes, y_pred) == pytest.approx(0.2)

    zeroes = zeroes.reshape((2, 5))
    y_pred = torch.stack([torch.zeros(5), torch.ones(5)])

    assert acc_metric.evaluate(zeroes, y_pred) == pytest.approx(0.5)


def test_accuracy_invalid():
    manager = MetricManager()
    acc_metric = manager.get_evaluation_metric("AccuracyMetric")

    zeroes = torch.zeros(5)
    y_pred = torch.zeros(4)

    with pytest.raises(TypeError):
        acc_metric.evaluate(zeroes, y_pred)
