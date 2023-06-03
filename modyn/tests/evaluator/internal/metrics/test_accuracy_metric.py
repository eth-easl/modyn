import numpy as np
import pytest
import torch
from modyn.evaluator.internal.metric_manager import MetricManager


def test_accuracy_metric():
    manager = MetricManager()
    acc_metric = manager.get_evaluation_metric("AccuracyMetric")

    assert acc_metric.progressive

    zeroes = torch.zeros(1)
    y_pred = torch.zeros(1)

    acc_metric.evaluate_batch(zeroes, y_pred, 1)

    assert acc_metric.get_evaluation_result() == 1

    zeroes = torch.zeros((2, 10))
    y_pred = torch.ones((2, 10))

    acc_metric.evaluate_batch(zeroes, y_pred, 2)

    assert acc_metric.get_evaluation_result() == pytest.approx(1.0 / 3)

    y_true = torch.from_numpy(np.array([0, 1, 0, 1, 0, 1]))
    y_pred = torch.from_numpy(np.array([1, 0, 1, 0, 0, 1]))

    acc_metric.evaluate_batch(y_true, y_pred, 6)

    assert acc_metric.get_evaluation_result() == pytest.approx(1.0 / 3)


def test_accuracy_metric_invalid():
    manager = MetricManager()
    acc_metric = manager.get_evaluation_metric("AccuracyMetric")

    zeroes = torch.zeros((5, 1))
    y_pred = torch.zeros((4, 1))

    with pytest.raises(TypeError):
        acc_metric.evaluate_batch(zeroes, y_pred, 5)
