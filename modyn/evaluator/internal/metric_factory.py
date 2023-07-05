from typing import Any

from modyn.evaluator.internal.metrics import AbstractEvaluationMetric, AbstractHolisticMetric, Accuracy, F1Score, RocAuc

all_metrics = {Accuracy, F1Score, RocAuc}


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
    def contains_holistic_metric(metrics: list[AbstractEvaluationMetric]) -> bool:
        for metric in metrics:
            if isinstance(metric, AbstractHolisticMetric):
                return True
        return False
