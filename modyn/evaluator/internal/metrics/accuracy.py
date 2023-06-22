from typing import Any

import torch
from modyn.evaluator.internal.metrics.abstract_decomposable_metric import AbstractDecomposableMetric


class Accuracy(AbstractDecomposableMetric):
    """
    Accuracy metric implementation.
    """

    def __init__(self, name: str, evaluation_transform_func: str, config: dict[str, Any]) -> None:
        super().__init__(name, evaluation_transform_func, config)
        self.seen_samples = 0
        self.total_correct = 0

    def batch_evaluated_callback(self, y_true: torch.tensor, y_pred: torch.tensor, batch_size: int) -> None:
        if self.evaluation_transformer_function:
            y_true, y_pred = self.evaluation_transformer_function(y_true, y_pred)
        if y_true.shape != y_pred.shape:
            raise TypeError(f"Shape of y_true and y_pred must match. Got {y_true.shape} and {y_pred.shape}.")
        assert y_true.shape[0] == batch_size, "Batch size and target label amount is not equal."

        # in order to use the same approach for scalar and vector case.
        if y_true.dim() < 2:
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)

        correct_labeled = torch.sum(torch.all(torch.eq(y_pred, y_true), dim=-1)).item()
        assert correct_labeled <= batch_size, "Total correct amount is greater than batch size."

        self.total_correct += correct_labeled
        self.seen_samples += batch_size

    def get_evaluation_result(self) -> float:
        if self.seen_samples == 0:
            self.warning("Did not see any samples.")
            return 0

        return float(self.total_correct) / self.seen_samples
