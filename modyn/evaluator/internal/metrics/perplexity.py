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
        # Validate y_pred dimensions.
        if y_pred.dim() < 2:
            raise RuntimeError("Invalid shape: y_pred must have at least 2 dimensions (batch, num_classes)")

        # Determine the expected number of tokens based on y_pred shape.
        if y_pred.dim() == 2:
            expected_tokens = y_pred.size(0)
        elif y_pred.dim() == 3:
            expected_tokens = y_pred.size(0) * y_pred.size(1)
        else:
            raise RuntimeError(f"y_pred with dimension {y_pred.dim()} not supported")

        if y_true.numel() != expected_tokens:
            raise RuntimeError("Mismatch in number of tokens between y_true and y_pred")

        # If any target token index is out of bounds, pad y_pred along the last dimension.
        num_classes = y_pred.size(-1)
        max_target = int(y_true.max().item())
        if max_target >= num_classes:
            padding_size = max_target + 1 - num_classes
            pad_tensor = torch.zeros(*y_pred.shape[:-1], padding_size, device=y_pred.device, dtype=y_pred.dtype)
            y_pred = torch.cat([y_pred, pad_tensor], dim=-1)

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
