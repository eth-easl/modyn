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
        assert y_pred.dim() in (2, 3), "y_pred must have 2 or 3 dimensions"
        if y_pred.dim() == 2:
            expected_tokens = y_pred.numel()
        elif y_pred.dim() == 3:
            expected_tokens = y_pred.size(0) * y_pred.size(1)
        if y_true.numel() != expected_tokens:  # pylint: disable=possibly-used-before-assignment
            print(f"y_true: {y_true.shape}, expected: {expected_tokens}, y_pred: {y_pred.shape}")
            raise RuntimeError("Mismatch in number of tokens between y_true and y_pred")

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        loss = loss_fn(
            y_pred.view(-1, y_pred.size(-1)),
            y_true.view(-1),
        )
        # count only the tokens that weren't ignored
        valid_tokens = (y_true.view(-1) != loss_fn.ignore_index).sum().item()

        self.total_loss += loss.item()
        self.total_tokens += valid_tokens

    def get_evaluation_result(self) -> float:
        if self.total_tokens == 0:
            self.warning("Did not see any samples.")
            return float("inf")
        return np.exp(self.total_loss / self.total_tokens)

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
