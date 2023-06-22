import pytest
from modyn.evaluator.internal.metric_factory import MetricFactory


def test_metric_unavailability():
    with pytest.raises(NotImplementedError):
        MetricFactory.get_evaluation_metric("UnknownMetric", config={}, evaluation_transform_func="")


def test_metric_availability():
    accuracy = MetricFactory.get_evaluation_metric("Accuracy", config={}, evaluation_transform_func="")
    f1_score = MetricFactory.get_evaluation_metric("F1-score", config={"test_attr": 10}, evaluation_transform_func="")
    auc_roc = MetricFactory.get_evaluation_metric("ROC-AUC", config={}, evaluation_transform_func="")

    assert accuracy.get_name() == "Accuracy"
    assert f1_score.get_name() == "F1-score"
    assert f1_score.config["test_attr"] == 10
    assert auc_roc.get_name() == "ROC-AUC"
