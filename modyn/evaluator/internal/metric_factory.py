from typing import overload

from modyn.config.schema.pipeline.evaluation.metrics import (
    MetricConfig,
    validate_metric_config_json,
)
from modyn.evaluator.internal.metrics import (
    AbstractEvaluationMetric,
    AbstractHolisticMetric,
    Accuracy,
    Bleu,
    F1Score,
    LLMEvaluation,
    Meteor,
    Perplexity,
    RocAuc,
)

all_metrics: set[type[AbstractEvaluationMetric]] = {Accuracy, F1Score, RocAuc, Perplexity, LLMEvaluation, Meteor, Bleu}


class MetricFactory:
    @overload
    @staticmethod
    def get_evaluation_metric(config: str) -> AbstractEvaluationMetric: ...

    @overload
    @staticmethod
    def get_evaluation_metric(config: MetricConfig) -> AbstractEvaluationMetric: ...

    @staticmethod
    def get_evaluation_metric(config: str | MetricConfig) -> AbstractEvaluationMetric:
        if isinstance(config, str):
            parsed_metric_config = validate_metric_config_json(config)
        else:
            parsed_metric_config = config

        for metric in all_metrics:
            if metric.__name__.lower() == parsed_metric_config.name.lower():
                return metric(parsed_metric_config)
        raise NotImplementedError(f"Metric {parsed_metric_config.name} is not available!")

    @staticmethod
    def prepare_metrics(metrics: dict[str, AbstractEvaluationMetric]) -> bool:
        contains_holistic = False
        for metric in metrics.values():
            metric.deserialize_evaluation_transformer()
            if isinstance(metric, AbstractHolisticMetric):
                contains_holistic = True
        return contains_holistic
