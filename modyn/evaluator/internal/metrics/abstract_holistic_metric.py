from abc import abstractmethod

import torch

from modyn.evaluator.internal.metrics.abstract_evaluation_metric import AbstractEvaluationMetric


class AbstractHolisticMetric(AbstractEvaluationMetric):
    """This abstract class is used to represent a holistic metric i.e a metric
    that is non-decomposable like median."""

    @abstractmethod
    def _dataset_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_samples: int) -> None:
        """Function that is called whenever the whole dataset can be evaluated.

        Args:
            y_true: True labels of the samples.
            y_pred: Model predictions.
            num_samples: Amount of elements that are evaluated.
        """
        raise NotImplementedError()

    def evaluate_dataset(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_samples: int) -> None:
        y_pred = self.transform_prediction(y_true, y_pred, num_samples)
        self._dataset_evaluated_callback(y_true, y_pred, num_samples)
