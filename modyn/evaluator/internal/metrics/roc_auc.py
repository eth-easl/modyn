import math

import torch

from modyn.config.schema.pipeline import RocAucMetricConfig
from modyn.evaluator.internal.metrics.abstract_holistic_metric import (
    AbstractHolisticMetric,
)


class RocAuc(AbstractHolisticMetric):
    """ROC-AUC metric implementation."""

    def __init__(self, config: RocAucMetricConfig) -> None:
        super().__init__(config)
        self.evaluation_result: float | None = None

    # Taken from
    # https://github.com/NVIDIA/DeepLearningExamples/blob/678b470fd78e0fdb84b3173bc25164d766e7821f/PyTorch/Recommendation/DLRM/dlrm/scripts/utils.py#L289
    # pylint: disable=unused-argument
    def _dataset_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_samples: int) -> None:
        """Calculate the ROC-AUC score for the given model output and ground
        truth.

        Args:
            y_true: The true values.
            y_pred: The predicted values by the model.
        """
        assert self.evaluation_result is None

        if y_true.dim() < 1 or len(y_true) < 2:
            # if the number of elements in y_pred or y_true is less than 2, then the ROC-AUC score is undefined
            self.evaluation_result = 0
            return

        y_true.squeeze_()
        y_pred.squeeze_()
        if y_true.shape != y_pred.shape:
            raise TypeError(f"Shape of y_true and y_pred must match. Got {y_true.shape} and {y_pred.shape}.")
        desc_score_indices = torch.argsort(y_pred, descending=True)
        y_score = y_pred[desc_score_indices]
        y_true = y_true[desc_score_indices]
        # we only need to squeeze the second dimension;
        # otherwise if there is only one non-zero element in (y_score[1:] - y_score[:-1]),
        # after squeezing it will become a scalar, which will cause an error in torch.cat
        distinct_value_indices = torch.nonzero(y_score[1:] - y_score[:-1], as_tuple=False).squeeze(dim=1)
        threshold_idxs = torch.cat([distinct_value_indices, torch.tensor([y_true.numel() - 1])])
        tps = torch.cumsum(y_true, dim=0)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        tps = torch.cat([torch.zeros(1), tps])
        fps = torch.cat([torch.zeros(1), fps])
        fpr = fps / fps[-1]
        tpr = tps / tps[-1]
        area = torch.trapz(tpr, fpr).item()
        if math.isnan(area):
            self.warning("Value of the area is NaN.")
            self.evaluation_result = 0
        else:
            self.evaluation_result = area

    def get_evaluation_result(self) -> float:
        assert self.evaluation_result is not None

        return self.evaluation_result

    def get_name(self) -> str:
        return "ROC-AUC"
