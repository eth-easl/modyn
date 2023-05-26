from modyn.evaluator.internal.metrics import AbstractEvaluationMetric
from modyn.utils import dynamic_module_import


class MetricManager:
    def get_evaluation_metric(self, metric_name: str) -> AbstractEvaluationMetric:
        evaluation_module = dynamic_module_import("modyn.evaluator.internal.metrics")
        if not hasattr(evaluation_module, metric_name):
            raise NotImplementedError(f"Metric {metric_name} not available!")

        metric_handler = getattr(evaluation_module, metric_name)

        return metric_handler()
