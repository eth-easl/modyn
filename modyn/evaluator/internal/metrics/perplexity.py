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

        assert y_pred.dim() in (2, 3), "y_pred must have 2 or 3 dimensions"
        
        # Reshape logits and labels
        logits = y_pred.view(-1, y_pred.size(-1))  # [batch*seq_len, vocab]
        labels = y_true.view(-1)                   # [batch*seq_len]
        
        # Pad y_true with -100 if it's shorter
        if logits.size(0) > labels.size(0):
            diff = logits.size(0) - labels.size(0)
            pad_tensor = torch.full((diff,), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([pad_tensor,labels], dim=0)
        elif logits.size(0) < labels.size(0):
            raise RuntimeError(f"y_true is longer than y_pred: {labels.size(0)} > {logits.size(0)}")

        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        loss = loss_fn(logits, labels)

        # Count valid tokens
        valid_tokens = (labels != -100).sum().item()

        self.total_loss += loss.item()
        self.total_tokens += valid_tokens


    def get_evaluation_result(self) -> float:
        if self.total_tokens == 0:
            self.warning("Did not see any samples.")
            return float("inf")
        return -np.exp(self.total_loss / self.total_tokens)

    def get_name(self) -> str:
        return "Perplexity"

    def transform_prediction(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_elements: int) -> torch.Tensor:
        """Checks whether the label and prediction values match in dimensions
        and that they contain num_elements elements. Additionally, transform
        the model output if needed.

        Args:
            y_true: true labels.
            y_pred: predicted labels/values.
            num_elements: the number of elements expected to compare.

        Returns:
            torch.Tensor: the (possibly transformed) model output.
        """
        if self.evaluation_transformer_function:
            y_pred = self.evaluation_transformer_function(y_pred)

        assert y_pred.shape[0] == num_elements, "Batch size and target label amount is not equal."

        return y_pred
