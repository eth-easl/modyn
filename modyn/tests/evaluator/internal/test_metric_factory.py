import pytest
from modyn.evaluator.internal.metric_factory import (
    ACCURACY_METRIC_NAME,
    F1_SCORE_METRIC_NAME,
    ROC_AUC_METRIC_NAME,
    MetricFactory,
)


def test_metric_unavailability():
    with pytest.raises(NotImplementedError):
        MetricFactory.get_evaluation_metric("UnknownMetric", config={}, evaluation_transform_func="")


def test_metric_availability():
    accuracy = MetricFactory.get_evaluation_metric(ACCURACY_METRIC_NAME, config={}, evaluation_transform_func="")
    f1_score = MetricFactory.get_evaluation_metric(
        F1_SCORE_METRIC_NAME, config={"test_attr": 10}, evaluation_transform_func=""
    )
    auc_roc = MetricFactory.get_evaluation_metric(ROC_AUC_METRIC_NAME, config={}, evaluation_transform_func="")

    assert accuracy.name == ACCURACY_METRIC_NAME
    assert f1_score.name == F1_SCORE_METRIC_NAME
    assert f1_score.config["test_attr"] == 10
    assert auc_roc.name == ROC_AUC_METRIC_NAME
