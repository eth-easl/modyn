import pytest
from pydantic import ValidationError

from modyn.evaluator.internal.metric_factory import MetricFactory


def test_metric_unavailability() -> None:
    with pytest.raises(ValidationError):
        MetricFactory.get_evaluation_metric('{"name": "UnknownMetric"}')


def test_metric_availability() -> None:
    accuracy = MetricFactory.get_evaluation_metric('{"name": "Accuracy"}')
    f1_score = MetricFactory.get_evaluation_metric('{"name": "F1Score", "num_classes": 10}')
    auc_roc = MetricFactory.get_evaluation_metric('{"name": "RocAuc"}')

    assert accuracy.get_name() == "Accuracy"
    assert f1_score.get_name() == "F1-macro"
    assert f1_score.config.num_classes == 10
    assert auc_roc.get_name() == "ROC-AUC"
