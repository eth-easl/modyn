from typing import Any

from modyn.evaluator.internal.metrics import AbstractEvaluationMetric, AbstractHolisticMetric, Accuracy, F1Score, RocAuc

ACCURACY_METRIC_NAME = "Accuracy"
F1_SCORE_METRIC_NAME = "F1-score"
ROC_AUC_METRIC_NAME = "ROC-AUC"

all_metrics = {ACCURACY_METRIC_NAME: Accuracy, F1_SCORE_METRIC_NAME: F1Score, ROC_AUC_METRIC_NAME: RocAuc}


class MetricFactory:
    @staticmethod
    def get_evaluation_metric(
        name: str, evaluation_transform_func: str, config: dict[str, Any]
    ) -> AbstractEvaluationMetric:
        if name not in all_metrics:
            raise NotImplementedError(f"Metric {name} is not available!")
        return all_metrics[name](name, evaluation_transform_func, config)

    @staticmethod
    def contains_holistic_metric(metrics: list[AbstractEvaluationMetric]) -> bool:
        for metric in metrics:
            if isinstance(metric, AbstractHolisticMetric):
                return True
        return False
