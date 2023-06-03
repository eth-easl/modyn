import torch
from modyn.evaluator.internal.metrics.abstract_evaluation_metric import AbstractEvaluationMetric


class AccuracyMetric(AbstractEvaluationMetric):
    """
    Accuracy metric implementation.
    """

    def __init__(self) -> None:
        super().__init__(True)
        self.seen_samples = 0
        self.total_correct = 0

    def evaluate_batch(self, y_true: torch.tensor, y_pred: torch.tensor, batch_size: int) -> None:
        if y_true.shape != y_pred.shape:
            raise TypeError(f"Shape of y_true and y_pred must match. Got {y_true.shape} and {y_pred.shape}.")
        assert y_true.shape[0] == batch_size, "Batch size and target label amount is not equal."

        if y_true.dim() < 2:
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)

        correct_labeled = torch.sum(torch.all(torch.eq(y_pred, y_true), dim=-1)).item()
        assert correct_labeled <= batch_size, "Total correct amount is greater than batch size."

        self.total_correct += correct_labeled
        self.seen_samples += batch_size

    def get_evaluation_result(self) -> float:
        if self.seen_samples == 0:
            return 0

        return float(self.total_correct) / self.seen_samples
