from abc import abstractmethod

import torch

from modyn.evaluator.internal.metrics.abstract_evaluation_metric import AbstractEvaluationMetric


class AbstractDecomposableMetric(AbstractEvaluationMetric):
    """This abstract class is used to represent a decomposable metric like
    accuracy."""

    @abstractmethod
    def _batch_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, batch_size: int) -> None:
        """Function that is called whenever a batch can be evaluated. Use it
        for bookkeeping and to store temporary results needed for the
        evaluation result.

        Args:
            y_true: True labels of the samples.
            y_pred: Model predictions.
            batch_size: Size of the batch.
        """
        raise NotImplementedError()

    def evaluate_batch(self, y_true: torch.Tensor, y_pred: torch.Tensor, batch_size: int) -> None:
        y_pred = self.transform_prediction(y_true, y_pred, batch_size)
        self._batch_evaluated_callback(y_true, y_pred, batch_size)
