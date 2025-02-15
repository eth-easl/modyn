import torch

from modyn.config.schema.pipeline import TwikiF1MetricConfig
from modyn.evaluator.internal.metrics.abstract_decomposable_metric import AbstractDecomposableMetric


class TwikiF1Score(AbstractDecomposableMetric):
    """TWIKI-Probes F1 Score implementation."""

    def __init__(self, config: TwikiF1MetricConfig) -> None:
        super().__init__(config)
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    # pylint: disable=unused-argument
    def _batch_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, batch_size: int) -> None:
        correct = torch.eq(y_true, y_pred)
        self.true_positives += correct.sum().item()
        self.false_positives += (~correct).sum().item()
        self.false_negatives += (~correct).sum().item()

    def get_evaluation_result(self) -> float:
        if self.true_positives == 0:
            return 0.0
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def get_name(self) -> str:
        return "TWIKI-Probes F1 Score"
