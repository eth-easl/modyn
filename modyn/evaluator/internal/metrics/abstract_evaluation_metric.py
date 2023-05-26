from abc import ABC, abstractmethod

import torch


class AbstractEvaluationMetric(ABC):
    """
    This abstract class is used to represent an evaluation metric which can be used to evaluate a trained model.
    """

    @abstractmethod
    def evaluate(self, y_true: torch.tensor, y_pred: torch.tensor) -> float:
        """
        Function implementing the metric.

        Args:
            y_true: True labels of the samples.
            y_pred: Model predictions.

        Returns:
            float: the calculated value of the metric.
        """
        raise NotImplementedError()
