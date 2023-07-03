import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from modyn.utils import EVALUATION_TRANSFORMER_FUNC_NAME, deserialize_function

logger = logging.getLogger(__name__)


class AbstractEvaluationMetric(ABC):
    """
    This abstract class is used to represent an evaluation metric which can be used to evaluate a trained model.
    """

    def __init__(self, evaluation_transformer: str, config: dict[str, Any]):
        """
        Initialize the evaluation metric.

        Args:
            evaluation_transformer: transformation that is applied to the label and model output before evaluation.
            config: configuration options for the metric.
        """
        self.config = config

        self.evaluation_transformer_function = deserialize_function(
            evaluation_transformer, EVALUATION_TRANSFORMER_FUNC_NAME
        )

    @abstractmethod
    def get_evaluation_result(self) -> float:
        """
        Get the final evaluation result.

        Returns:
            float: the calculated value of the metric.
        """
        raise NotImplementedError()

    def warning(self, message: str) -> None:
        logger.warning(f"[{self.get_name()}] {message}")

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """
        Get the name of the metric.

        Returns:
            str: the metric name.
        """
        raise NotImplementedError()

    def transform_prediction(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_elements: int) -> torch.Tensor:
        """
        Checks whether the label and prediction values match in dimensions and that they contain num_elements elements.
        Additionally, transform the model output if needed

        Args:
            y_true: true labels.
            y_pred: predicted labels/values.
            num_elements: the number of elements expected to compare.

        Returns:
            torch.Tensor: the (possibly transformed) model output.
        """
        if self.evaluation_transformer_function:
            y_pred = self.evaluation_transformer_function(y_pred)
        if y_true.shape != y_pred.shape:
            raise TypeError(f"Shape of y_true and y_pred must match. Got {y_true.shape} and {y_pred.shape}.")
        assert y_pred.shape[0] == num_elements, "Batch size and target label amount is not equal."

        return y_pred
