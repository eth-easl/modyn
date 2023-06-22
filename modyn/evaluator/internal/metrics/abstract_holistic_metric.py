from abc import abstractmethod

import torch
from modyn.evaluator.internal.metrics.abstract_evaluation_metric import AbstractEvaluationMetric


class AbstractHolisticMetric(AbstractEvaluationMetric):
    """
    This abstract class is used to represent a holistic metric i.e a metric that is non-decomposable like median.
    """

    @abstractmethod
    def dataset_evaluated_callback(self, y_true: torch.tensor, y_pred: torch.tensor) -> None:
        """
        Function that is called whenever the whole dataset can be evaluated.

        Args:
            y_true: True labels of the samples.
            y_pred: Model predictions.
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
