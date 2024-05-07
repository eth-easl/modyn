from typing import Any

from modyn.evaluator.internal.metrics import AbstractEvaluationMetric, AbstractHolisticMetric, Accuracy, F1Score, RocAuc

all_metrics: set[type[AbstractEvaluationMetric]] = {Accuracy, F1Score, RocAuc}


class MetricFactory:
    @staticmethod
    def get_evaluation_metric(
        name: str, evaluation_transform_func: str, config: dict[str, Any]
    ) -> AbstractEvaluationMetric:
        for metric in all_metrics:
            if metric.get_name() == name:
                return metric(evaluation_transform_func, config)
        raise NotImplementedError(f"Metric {name} is not available!")

    @staticmethod
    def prepare_metrics(metrics: list[AbstractEvaluationMetric]) -> bool:
        contains_holistic = False
        for metric in metrics:
            metric.deserialize_evaluation_transformer()
            if isinstance(metric, AbstractHolisticMetric):
                return True
        return contains_holistic
