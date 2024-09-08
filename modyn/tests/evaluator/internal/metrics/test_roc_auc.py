import numpy as np
import pytest
import torch

from modyn.config.schema.pipeline import RocAucMetricConfig
from modyn.evaluator.internal.metrics import AbstractHolisticMetric, RocAuc


def get_evaluation_result(y_true: np.ndarray, y_score: np.ndarray):
    roc_auc = RocAuc(RocAucMetricConfig())
    roc_auc.evaluate_dataset(torch.from_numpy(y_true), torch.from_numpy(y_score), y_true.shape[0])

    return roc_auc.get_evaluation_result()


def test_roc_auc_metric():
    roc_auc = RocAuc(RocAucMetricConfig())
    assert isinstance(roc_auc, AbstractHolisticMetric)

    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.arange(0.1, 1, 0.1)
    assert get_evaluation_result(y_true, y_score) == pytest.approx(1)

    y_true[4] = 1
    assert get_evaluation_result(y_true, y_score) == pytest.approx(1)

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    assert get_evaluation_result(y_true, y_score) == pytest.approx(0)

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    assert get_evaluation_result(y_true, y_score) == pytest.approx(0.5)

    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    assert get_evaluation_result(y_true, y_score) == pytest.approx(1)

    y_true = np.array([0, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.7, 0.3, 0.4])

    assert get_evaluation_result(y_true, y_score) == pytest.approx(2 / 3)


def test_roc_auc_edge_cases():
    roc_auc = RocAuc(RocAucMetricConfig())
    assert isinstance(roc_auc, AbstractHolisticMetric)

    y_true = np.array([0])
    y_score = np.arange(0.1)
    assert get_evaluation_result(y_true, y_score) == 0

    y_true = np.array([])
    y_score = np.array([])
    assert get_evaluation_result(y_true, y_score) == 0


def test_roc_auc_with_two_entries():
    y_true = np.array([0, 1])
    y_score = np.array([0.1, 0.6])
    # this is to test that we correctly squeeze the dimension in _dataset_evaluated_callback()
    # we expect no exception
    get_evaluation_result(y_true, y_score)


def test_roc_auc_invalid():
    with pytest.raises(TypeError):
        get_evaluation_result(np.array([1, 1, 1]), np.array([0.2, 0.3]))

    y_true = np.zeros(5)
    y_score = np.array([0.1, 0.1, 0.2, 0.5, 0.4])

    assert get_evaluation_result(y_true, y_score) == 0
