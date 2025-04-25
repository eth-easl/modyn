import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from modyn.config.schema.pipeline import MetricConfig
from modyn.utils.utils import instantiate_class

logger = logging.getLogger(__name__)


class AbstractEvaluationMetric(ABC):
    """This abstract class is used to represent an evaluation metric which can
    be used to evaluate a trained model."""

    def __init__(self, config: MetricConfig):
        """Initialize the evaluation metric.

        Args:
            config: Configuration for the metric.
        """
        self.config = config
        self.evaluation_transformer_function: Callable[[torch.Tensor], torch.Tensor] | None = None
        if config.tokenizer is not None:
            self._tokenizer = (
                instantiate_class("modyn.models.tokenizers", config.tokenizer) if config.tokenizer else None
            )
        self.requires_generation = config.requires_generation

    def deserialize_evaluation_transformer(self) -> None:
        """Deserialize the evaluation transform function."""
        self.evaluation_transformer_function = self.config.evaluation_transformer_function_deserialized  # type: ignore

    @abstractmethod
    def get_evaluation_result(self) -> float:
        """Get the final evaluation result.

        Returns:
            float: the calculated value of the metric.
        """
        raise NotImplementedError()

    def warning(self, message: str) -> None:
        logger.warning(f"[{self.get_name()}] {message}")

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the metric.

        Returns:
            str: the metric name.
        """
        raise NotImplementedError()

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
        if self.config.shape_check and y_true.shape != y_pred.shape:
            raise TypeError(f"Shape of y_true and y_pred must match. Got {y_true.shape} and {y_pred.shape}.")
        assert y_pred.shape[0] == num_elements, "Batch size and target label amount is not equal."

        return y_pred

    def decode_ids(self, token_ids: torch.Tensor) -> str:
        """
        Turn a tensor of token-IDs into a single string, using the
        tokenizer if available, otherwise just join the IDs.
        """
        ids = token_ids.tolist()
        # If a tokenizer was set up on this metric, use it:
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            return self._tokenizer.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        # Fallback: space-join the raw IDs
        return " ".join(str(i) for i in ids)
