import torch
from modyn.evaluator.internal.metrics.abstract_evaluation_metric import AbstractEvaluationMetric


class AccuracyMetric(AbstractEvaluationMetric):
    """
    Accuracy metric implementation.
    """

    def evaluate(self, y_true: torch.tensor, y_pred: torch.tensor) -> float:
        if y_true.shape != y_pred.shape:
            raise TypeError(f"Shape of y_true and y_pred must match. Got {y_true.shape} and {y_pred.shape}.")

        return (torch.sum(torch.eq(y_pred, y_true)) / y_true.numel()).item()
