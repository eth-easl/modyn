import nltk
import torch
from nltk.translate.meteor_score import meteor_score

from modyn.config.schema.pipeline import MeteorMetricConfig
from modyn.evaluator.internal.metrics import AbstractDecomposableMetric

nltk.download("wordnet", quiet=True)


class Meteor(AbstractDecomposableMetric):
    def __init__(self, config: MeteorMetricConfig) -> None:
        super().__init__(config)
        self.evaluation_result: float | None = None

    def _batch_evaluated_callback(self, y_true: torch.tensor, y_pred: torch.tensor, num_samples: int) -> None:  # pylint: disable=unused-argument
        # Support both tensors (token IDs) and raw strings
        refs = [self.decode_ids(ref) for ref in y_true]
        preds = [self.decode_ids(pred) for pred in y_pred]

        scores = [meteor_score([r.split()], p.split()) for r, p in zip(refs, preds)]
        self.evaluation_result = sum(scores) / len(scores)

    def get_evaluation_result(self) -> float:
        if self.evaluation_result is None:
            self.warning("No METEOR scores computed.")
            return 0.0
        return self.evaluation_result

    def get_name(self) -> str:
        return "METEOR"
