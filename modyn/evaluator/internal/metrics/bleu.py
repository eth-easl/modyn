import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from modyn.config.schema.pipeline.evaluation.metrics import BleuMetricConfig
from modyn.evaluator.internal.metrics import AbstractHolisticMetric


class BleuScore(AbstractHolisticMetric):
    def __init__(self, config: BleuMetricConfig):
        super().__init__(config)
        self._all_references: list[list[list[str]]] = []
        self._all_hypotheses: list[list[str]] = []

    def _dataset_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_samples: int) -> None:  # pylint: disable=unused-argument
        # decode & tokenize, but store them
        for ref_ids, hyp_ids in zip(y_true, y_pred):
            ref_tokens = self.decode_ids(ref_ids).split()
            hyp_tokens = self.decode_ids(hyp_ids).split()
            self._all_references.append([ref_tokens])  # outer list for multi‐ref
            self._all_hypotheses.append(hyp_tokens)

    def get_evaluation_result(self) -> float:
        if not self._all_hypotheses:
            self.warning("No BLEU scores computed.")
            return 0.0
        # this uses the *entire* corpus
        chencherry = SmoothingFunction()
        return corpus_bleu(self._all_references, self._all_hypotheses, smoothing_function=chencherry.method1)

    def get_name(self) -> str:
        return "BLEU Score"
