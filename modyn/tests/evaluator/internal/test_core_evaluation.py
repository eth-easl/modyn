import pytest
from pydantic import ValidationError

from modyn.config import F1ScoreMetricConfig
from modyn.config.schema.pipeline import AccuracyMetricConfig
from modyn.evaluator.internal.core_evaluation import setup_metrics
from modyn.evaluator.internal.metrics.accuracy import Accuracy
from modyn.evaluator.internal.metrics.f1_score import F1Score


def test_setup_metrics():
    acc_metric_config = AccuracyMetricConfig().model_dump_json()
    metrics = setup_metrics([acc_metric_config])

    assert len(metrics) == 1
    assert metrics["Accuracy"].get_name() == "Accuracy"
    unknown_metric_config = '{"name": "UnknownMetric", "config": "", "evaluation_transformer_function": ""}'
    with pytest.raises(ValidationError):
        setup_metrics([unknown_metric_config])

    f1score_metric_config = F1ScoreMetricConfig(num_classes=2).model_dump_json()
    metrics = setup_metrics([acc_metric_config, acc_metric_config, f1score_metric_config])
    assert len(metrics) == 2
    assert isinstance(metrics["Accuracy"], Accuracy)
    assert isinstance(metrics["F1-macro"], F1Score)


def test__setup_metrics_multiple_f1():
    macro_f1_config = F1ScoreMetricConfig(
        evaluation_transformer_function="",
        num_classes=2,
        average="macro",
    ).model_dump_json()

    micro_f1_config = F1ScoreMetricConfig(
        evaluation_transformer_function="",
        num_classes=2,
        average="micro",
    ).model_dump_json()

    # not double macro, but macro and micro work
    metrics = setup_metrics([macro_f1_config, micro_f1_config, macro_f1_config])

    assert len(metrics) == 2
    assert isinstance(metrics["F1-macro"], F1Score)
    assert isinstance(metrics["F1-micro"], F1Score)
    assert metrics["F1-macro"].config.average == "macro"
    assert metrics["F1-micro"].config.average == "micro"
    assert metrics["F1-macro"].get_name() == "F1-macro"
    assert metrics["F1-micro"].get_name() == "F1-micro"
