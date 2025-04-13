import logging

import torch

from modyn.config.schema.pipeline import MetricConfig
from modyn.evaluator.internal.metrics import AbstractEvaluationMetric
from modyn.utils import (
    instantiate_class,
)

logger = logging.getLogger(__name__)


class AbstractTextMetric(AbstractEvaluationMetric):
    """Abstract text metric that directly uses a tokenizer to decode token IDs."""

    def __init__(self, config: MetricConfig) -> None:
        super().__init__(config)
        self._tokenizer = (
            instantiate_class("modyn.models.tokenizers", config.tokenizer, max_token_length=config.seq_length)
            if config.tokenizer
            else None
        )

    def decode_ids(self, token_ids_tensor: torch.Tensor) -> str:
        """Decode a single torch.Tensor of token IDs into a text string via the tokenizer or fallback."""
        token_ids = token_ids_tensor.tolist()
        if self._tokenizer is not None:
            return self._tokenizer.tokenizer.decode(token_ids)
        # Fallback: just space-join the IDs if no tokenizer is provided
        return " ".join(str(t) for t in token_ids)

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
