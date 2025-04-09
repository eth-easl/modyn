import nltk
import torch
from nltk.translate.meteor_score import meteor_score

from modyn.config.schema.pipeline import MeteorMetricConfig
from modyn.evaluator.internal.metrics import AbstractTextMetric

nltk.download("wordnet", quiet=True)


class Meteor(AbstractTextMetric):
    def __init__(self, config: MeteorMetricConfig, tokenizer: str) -> None:
        super().__init__(config, tokenizer)
        self.evaluation_result: float | None = None

    def _dataset_evaluated_callback(self, y_true: torch.tensor, y_pred: torch.tensor, num_samples: int) -> None:
        assert self.evaluation_result is None

        if not y_true or not y_pred or len(y_true) != len(y_pred):
            self.evaluation_result = 0
            return

        # Support both tensors (token IDs) and raw strings
        refs = [self.decode_ids(ref) if hasattr(ref, "tolist") else ref for ref in y_true]
        preds = [self.decode_ids(pred) if hasattr(pred, "tolist") else pred for pred in y_pred]

        scores = [meteor_score([r.split()], p.split()) for r, p in zip(refs, preds)]
        self.evaluation_result = sum(scores) / len(scores)

    def get_evaluation_result(self) -> float:
        assert self.evaluation_result is not None
        return self.evaluation_result

    def get_name(self) -> str:
        return "METEOR"
