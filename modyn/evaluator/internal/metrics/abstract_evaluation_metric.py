from abc import ABC, abstractmethod

import torch


class AbstractEvaluationMetric(ABC):
    """
    This abstract class is used to represent an evaluation metric which can be used to evaluate a trained model.
    """

    def __init__(self, progressive: bool):
        """
        Abstract class for all evaluation metrics.
        Currently we only allow progressive metrics i.e metrics where the final evaluation value can be calculated
        on-the-fly without having the model output from all batches.

        Args:
            progressive: whether the metrics is progressive.
        """
        self.progressive = progressive

        if not self.progressive:
            raise NotImplementedError("Only progressive metrics are allowed.")

    @abstractmethod
    def evaluate_batch(self, y_true: torch.tensor, y_pred: torch.tensor, batch_size: int) -> None:
        """
        Function that is called whenever a batch was evaluated.
        Use it for bookkeeping and to store temporary results needed for the final evaluation

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
