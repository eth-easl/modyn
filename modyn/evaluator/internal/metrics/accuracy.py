import torch

from modyn.config.schema.pipeline import AccuracyMetricConfig
from modyn.evaluator.internal.metrics.abstract_decomposable_metric import AbstractDecomposableMetric


class Accuracy(AbstractDecomposableMetric):
    """Accuracy metric implementation."""

    def __init__(self, config: AccuracyMetricConfig) -> None:
        super().__init__(config)
        self.samples_seen = 0
        self.total_correct = 0

    # pylint: disable=unused-argument
    def _batch_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, batch_size: int) -> None:
        if self.config.topn == 1:
            labeled_correctly = torch.sum(torch.eq(y_pred, y_true)).item()
        else:
            # For top n accuracy, the evaluation_transform_func
            # may NOT do an argmax! This is just a transformation for the n = 1 case.
            _, top_pred_indices = torch.topk(y_pred, self.config.topn, dim=1)
            labeled_correctly = torch.sum(top_pred_indices == y_true.unsqueeze(1)).item()

        self.total_correct += labeled_correctly
        self.samples_seen += batch_size

    def get_evaluation_result(self) -> float:
        if self.samples_seen == 0:
            self.warning("Did not see any samples.")
            return 0

        return float(self.total_correct) / self.samples_seen

    def get_name(self) -> str:
        return "Accuracy" if self.config.topn == 1 else f"Top-{self.config.topn}-Accuracy"
