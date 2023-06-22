import logging
from abc import ABC, abstractmethod
from typing import Any

from modyn.utils import EVALUATION_TRANSFORMER_FUNC_NAME, deserialize_function

logger = logging.getLogger(__name__)


class AbstractEvaluationMetric(ABC):
    """
    This abstract class is used to represent an evaluation metric which can be used to evaluate a trained model.
    """

    def __init__(self, name: str, evaluation_transformer: str, config: dict[str, Any]):
        """
        Initialize the evaluation metric.

        Args:
            name: the name of the metric.
            evaluation_transformer: transformation that is applied to the label and model output before evaluation.
            config: configuration options for the metric.
        """
        self.name = name
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
        logger.warning(f"[{self.name}] {message}")
