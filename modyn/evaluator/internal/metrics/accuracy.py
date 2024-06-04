from typing import Any

import torch
from modyn.evaluator.internal.metrics.abstract_decomposable_metric import AbstractDecomposableMetric


class Accuracy(AbstractDecomposableMetric):
    """
    Accuracy metric implementation.
    """

    def __init__(self, evaluation_transform_func: str, config: dict[str, Any]) -> None:
        super().__init__(evaluation_transform_func, config)
        self.samples_seen = 0
        self.total_correct = 0

    def _batch_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, batch_size: int) -> None:
        labeled_correctly = torch.sum(torch.eq(y_pred, y_true)).item()

        self.total_correct += labeled_correctly
        self.samples_seen += batch_size

    def get_evaluation_result(self) -> float:
        if self.samples_seen == 0:
            self.warning("Did not see any samples.")
            return 0

        return float(self.total_correct) / self.samples_seen

    def get_name(self) -> str:
        return "Accuracy"
