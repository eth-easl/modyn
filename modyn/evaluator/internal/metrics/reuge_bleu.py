import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from modyn.config.schema.pipeline import BleuMetricConfig, RougeMetricConfig

from .abstract_text_metric import AbstractTextMetric


class Bleu(AbstractTextMetric):
    """BLEU Score metric for text generation evaluation."""

    def __init__(self, config: BleuMetricConfig, tokenizer: str) -> None:
        super().__init__(config, tokenizer)
        self.bleu_scores: list[float] = []

    def _dataset_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_samples: int) -> None:  # pylint: disable=unused-argument
        chencherry = SmoothingFunction()
        for ref_ids, hyp_ids in zip(y_true, y_pred):
            ref_text = self._tokenizer.tokenizer.decode(ref_ids)
            hyp_text = self._tokenizer.tokenizer.decode(hyp_ids)

            # Convert to lists of tokens
            ref_tokens = ref_text.split()
            hyp_tokens = hyp_text.split()
            print(f"Reference: {ref_tokens}")
            print(f"Hypothesis: {hyp_tokens}")
            references = [ref_tokens]  # BLEU expects a list of references
            score = sentence_bleu(references, hyp_tokens, smoothing_function=chencherry.method1)
            self.bleu_scores.append(score)
            print(f"Sample BLEU: {score}")

    def get_evaluation_result(self) -> float:
        if not self.bleu_scores:
            self.warning("No BLEU scores computed.")
            return 0.0
        return float(np.mean(self.bleu_scores))

    def get_name(self) -> str:
        return "BLEU Score"


class ROUGEScore(AbstractTextMetric):
    """ROUGE Score metric for text evaluation."""

    def __init__(self, config: RougeMetricConfig, tokenizer: str) -> None:
        super().__init__(config, tokenizer)
        self.scores: dict[str, list[float]] = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

    def _dataset_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_samples: int) -> None:  # pylint: disable=unused-argument
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        for ref_ids, hyp_ids in zip(y_true, y_pred):
            ref_text = self.decode_ids(ref_ids)
            hyp_text = self.decode_ids(hyp_ids)
            result = scorer.score(ref_text, hyp_text)
            self.scores["rouge-1"].append(result["rouge1"].fmeasure)
            self.scores["rouge-2"].append(result["rouge2"].fmeasure)
            self.scores["rouge-l"].append(result["rougeL"].fmeasure)

    def get_evaluation_result(self) -> float:
        if not self.scores["rouge-1"]:
            self.warning("No ROUGE scores computed.")
            return 0.0
        return (
            float(
                (np.mean(self.scores["rouge-1"]))
                + float(np.mean(self.scores["rouge-2"]))
                + float(np.mean(self.scores["rouge-l"]))
            )
            / 3
        )

    def get_name(self) -> str:
        return "ROUGE Score"
