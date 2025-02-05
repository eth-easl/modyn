import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from modyn.config.schema.pipeline import BleuMetricConfig, RougeMetricConfig
from modyn.evaluator.internal.metrics.abstract_holistic_metric import AbstractHolisticMetric


class BLEUScore(AbstractHolisticMetric):
    """BLEU Score metric implementation for text generation evaluation."""

    def __init__(self, config: BleuMetricConfig) -> None:
        super().__init__(config)
        self.bleu_scores: list[float] = []

    def _dataset_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_samples: int) -> None:
        """Computes BLEU score per sample and stores results."""
        chencherry = SmoothingFunction()
        for ref, hyp in zip(y_true, y_pred):
            reference = [ref.tolist()]  # BLEU expects a list of references
            hypothesis = hyp.tolist()
            score = sentence_bleu(reference, hypothesis, smoothing_function=chencherry.method1)
            self.bleu_scores.append(score)

    def get_evaluation_result(self) -> float:
        """Returns the average BLEU score across all evaluated samples."""
        if not self.bleu_scores:
            self.warning("No BLEU scores computed.")
            return 0.0
        return float(np.mean(self.bleu_scores))  # Explicit cast to float

    def get_name(self) -> str:
        return "BLEU Score"


class ROUGEScore(AbstractHolisticMetric):
    """ROUGE Score metric implementation for text evaluation."""

    def __init__(self, config: RougeMetricConfig) -> None:
        super().__init__(config)
        self.scores: dict[str, list[float]] = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

    def _dataset_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_samples: int) -> None:
        """Computes ROUGE scores for predictions."""
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        for ref, hyp in zip(y_true, y_pred):
            scores = scorer.score(ref.tolist(), hyp.tolist())
            self.scores["rouge-1"].append(scores["rouge1"].fmeasure)
            self.scores["rouge-2"].append(scores["rouge2"].fmeasure)
            self.scores["rouge-l"].append(scores["rougeL"].fmeasure)

    def get_evaluation_results(self) -> dict[str, float]:  # New method name to avoid superclass type conflict
        """Returns the averaged ROUGE scores."""
        if not self.scores["rouge-1"]:
            self.warning("No ROUGE scores computed.")
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        return {
            "rouge-1": float(np.mean(self.scores["rouge-1"])),
            "rouge-2": float(np.mean(self.scores["rouge-2"])),
            "rouge-l": float(np.mean(self.scores["rouge-l"])),
        }

    def get_name(self) -> str:
        return "ROUGE Score"
