import numpy as np
import torch

from modyn.evaluator.internal.metrics.abstract_holistic_metric import AbstractHolisticMetric


class GLUEScore(AbstractHolisticMetric):
    """GLUE Score metric implementation."""

    def __init__(self, config: GlueScoreMetricConfig) -> None:
        super().__init__(config)
        self.results = []

    def _dataset_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_samples: int) -> None:
        """Stores individual task scores to be averaged later."""
        self.results.append(torch.mean(y_pred).item())  # Example: replace with actual GLUE metric computation

    def get_evaluation_result(self) -> float:
        if not self.results:
            self.warning("No GLUE scores computed.")
            return 0.0
        return np.mean(self.results)

    def get_name(self) -> str:
        return "GLUE Score"
