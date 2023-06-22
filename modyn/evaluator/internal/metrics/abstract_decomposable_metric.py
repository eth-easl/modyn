from abc import abstractmethod

import torch
from modyn.evaluator.internal.metrics.abstract_evaluation_metric import AbstractEvaluationMetric


class AbstractDecomposableMetric(AbstractEvaluationMetric):
    """
    This abstract class is used to represent a decomposable metric like accuracy.
    """

    @abstractmethod
    def batch_evaluated_callback(self, y_true: torch.tensor, y_pred: torch.tensor, batch_size: int) -> None:
        """
        Function that is called whenever a batch can be evaluated.
        Use it for bookkeeping and to store temporary results needed for the evaluation result.

        Args:
            y_true: True labels of the samples.
            y_pred: Model predictions.
            batch_size: Size of the batch.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_evaluation_result(self) -> float:
        """
        Get the final evaluation result.

        Returns:
            float: the calculated value of the metric.
        """
        raise NotImplementedError()
