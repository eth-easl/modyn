from torchmetrics.text.bleu import BLEUScore
import torch

from modyn.config.schema.pipeline.evaluation.metrics import BleuMetricConfig
from modyn.evaluator.internal.metrics import AbstractHolisticMetric


class Bleu(AbstractHolisticMetric):
    def __init__(self, config: BleuMetricConfig):
        super().__init__(config)
        # 4‑gram BLEU with smoothing; matches corpus_bleu defaults
        self._metric = BLEUScore(n_gram=4, smooth=True)
        self._debug  = True

    @torch.no_grad()
    def _dataset_evaluated_callback(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, _
    ) -> None:
        refs = self._tokenizer.tokenizer.batch_decode(
            y_true.tolist(), skip_special_tokens=True
        )
        hyps = self._tokenizer.tokenizer.batch_decode(
            y_pred.tolist(), skip_special_tokens=True
        )

        # BLEUScore expects List[str] and List[List[str]]
        self._metric.update(hyps, [[r] for r in refs])

        if self._debug:
            print(f"[BLEU] running corpus score: {self._metric.compute():.4f}")

    def get_evaluation_result(self) -> float:
        try:
            return self._metric.compute().item()
        except (ValueError, RuntimeError):
            self.warning("No BLEU scores computed.")
            return 0.0

    def get_name(self) -> str:
        return "BLEU"
