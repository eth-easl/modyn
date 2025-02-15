import numpy as np
import torch

from modyn.config.schema.pipeline import PerplexityMetricConfig
from modyn.evaluator.internal.metrics.abstract_decomposable_metric import AbstractDecomposableMetric


class Perplexity(AbstractDecomposableMetric):
    """Standard Perplexity metric implementation."""

    def __init__(self, config: PerplexityMetricConfig) -> None:
        super().__init__(config)
        self.total_loss = 0.0
        self.total_tokens = 0

    def _batch_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, batch_size: int) -> None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fn(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1))
        self.total_loss += loss.item()
        self.total_tokens += y_true.numel()

    def get_evaluation_result(self) -> float:
        if self.total_tokens == 0:
            self.warning("Did not see any samples.")
            return float("inf")
        return np.exp(self.total_loss / self.total_tokens)

    def get_name(self) -> str:
        return "Perplexity"
