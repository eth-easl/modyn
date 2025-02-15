from nltk.translate.meteor_score import meteor_score

from modyn.config.schema.pipeline import MeteorMetricConfig
from modyn.evaluator.internal.metrics.abstract_holistic_metric import AbstractHolisticMetric


class Meteor(AbstractHolisticMetric):
    """METEOR metric implementation for text-based model evaluation."""

    def __init__(self, config: MeteorMetricConfig) -> None:
        super().__init__(config)
        self.evaluation_result: float | None = None

    def _dataset_evaluated_callback(self, y_true: list[str], y_pred: list[str], num_samples: int) -> None:
        """Calculate the METEOR score for the given model outputs and reference texts.

        Args:
            y_true: List of reference texts (ground truth).
            y_pred: List of predicted texts by the model.
        """
        assert self.evaluation_result is None

        if not y_true or not y_pred or len(y_true) != len(y_pred):
            self.evaluation_result = 0  # Undefined METEOR score in such cases
            return

        # Compute METEOR score for each pair of reference & prediction
        meteor_scores = [meteor_score([ref], pred) for ref, pred in zip(y_true, y_pred)]

        # Average METEOR score across all samples
        self.evaluation_result = sum(meteor_scores) / len(meteor_scores)

    def get_evaluation_result(self) -> float:
        assert self.evaluation_result is not None
        return self.evaluation_result

    def get_name(self) -> str:
        return "METEOR"
