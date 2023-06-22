from typing import Any

import torch
from modyn.evaluator.internal.metrics.abstract_decomposable_metric import AbstractDecomposableMetric


class F1Score(AbstractDecomposableMetric):
    """
    F1-score implementation.
    """

    def __init__(self, name: str, evaluation_transform_func: str, config: dict[str, Any]) -> None:
        super().__init__(name, evaluation_transform_func, config)
        self.seen_samples = 0
        self.total_correct = 0
        # TODO(#269): build confusion matrix

    def batch_evaluated_callback(self, y_true: torch.tensor, y_pred: torch.tensor, batch_size: int) -> None:
        if self.evaluation_transformer_function:
            y_true, y_pred = self.evaluation_transformer_function(y_true, y_pred)
        # TODO(#269): implement callback

    def get_evaluation_result(self) -> float:
        # TODO(#269): implement final result
        return 0.0
