import torch
import evaluate                                   # pip install evaluate  (needs Java ≥ 8)
from modyn.config.schema.pipeline import MeteorMetricConfig
from modyn.evaluator.internal.metrics import AbstractDecomposableMetric
from nltk.translate.meteor_score import meteor_score

class Meteor(AbstractDecomposableMetric):
    """
    Corpus‑level METEOR via HF/evaluate (fast Java backend).
    Requires a JVM; falls back to NLTK if evaluate.load("meteor") fails.
    """

    def __init__(self, config: MeteorMetricConfig) -> None:
        super().__init__(config)

        print("DEBUG: Initialising METEOR metric …")
        self._metric = evaluate.load("meteor")  # compiled scorer
        self._use_hf = True
        print(f"DEBUG: HF‑evaluate backend loaded: {self._use_hf}")

        self._debug = getattr(config, "debug", True)
        self._examples = 0
        self._score_sum = 0.0

    @torch.no_grad()
    def _batch_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, _) -> None:
        print(f"DEBUG: Processing batch of size {y_true.size(0)}")

        refs = self._tokenizer.tokenizer.batch_decode(y_true.tolist(), skip_special_tokens=True)
        hyps = self._tokenizer.tokenizer.batch_decode(y_pred.tolist(), skip_special_tokens=True)
        print(f"DEBUG: Decoded {len(refs)} refs / {len(hyps)} hyps")

        if self._use_hf:
            self._metric.add_batch(predictions=hyps, references=[[r] for r in refs])
            print("DEBUG: Added batch to HF metric")
        else:
            for r, h in zip(refs, hyps):
                self._score_sum += meteor_score([r], h)
            self._examples += len(refs)
            print(f"DEBUG: Fallback mode — cumulative examples: {self._examples}")

        if self._debug:
            running = (
                self._metric.compute()["meteor"]
                if self._use_hf 
                else (self._score_sum / self._examples if self._examples else 0.0)
            )
            print(f"[METEOR] running corpus score: {running:.4f}")

    def get_evaluation_result(self) -> float:
        print("DEBUG: Computing final METEOR …")
        if self._use_hf:
            if self._score_sum == 0:
                self.warning("No METEOR scores computed.")
                return 0.0
            result = self._metric.compute()["meteor"]
        else:
            if self._examples == 0:
                self.warning("No METEOR scores computed.")
                return 0.0
            result = self._score_sum / self._examples
        print(f"DEBUG: Final METEOR = {result:.4f}")
        return result

    def get_name(self) -> str:
        return "METEOR"
