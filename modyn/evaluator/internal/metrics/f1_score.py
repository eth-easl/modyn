import numpy as np
import torch
from modyn.config.schema.pipeline import F1ScoreMetricConfig
from modyn.evaluator.internal.metrics.abstract_decomposable_metric import AbstractDecomposableMetric


class F1Score(AbstractDecomposableMetric):
    """
    F1-score implementation. Configuration options:
    - num_classes: the total number of classes.
    - (optional) average: the method used to average f1-score in the multiclass setting (default macro).
    - (optional) pos_label: the positive label used in binary classification (default 1), only its f1-score is returned.
    """

    def __init__(self, config: F1ScoreMetricConfig) -> None:
        super().__init__(config)
        # store true positives (tp), false positives (fp) and false negatives (fn) in matrix
        self.classification_matrix = np.zeros((3, self.config.num_classes))

    def _batch_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, batch_size: int) -> None:
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        correct_mask = np.equal(y_true, y_pred)
        wrong_mask = np.invert(correct_mask)

        group_tp = np.unique(y_pred[correct_mask], return_counts=True)
        self.classification_matrix[0, group_tp[0]] += group_tp[1]

        group_fp = np.unique(y_pred[wrong_mask], return_counts=True)
        self.classification_matrix[1, group_fp[0]] += group_fp[1]

        group_fn = np.unique(y_true[wrong_mask], return_counts=True)
        self.classification_matrix[2, group_fn[0]] += group_fn[1]

    def get_evaluation_result(self) -> float:
        true_positives = self.classification_matrix[0]
        false_positives = self.classification_matrix[1]

        sum_tp: int = np.sum(true_positives)
        total_samples = sum_tp + np.sum(false_positives)

        if total_samples == 0:
            self.warning("Did not see any samples.")
            return 0

        # equivalent to accuracy
        if self.config.average == "micro":
            return sum_tp / total_samples

        false_negatives = self.classification_matrix[2]

        denominator = 2 * true_positives + false_positives + false_negatives
        numerator = 2 * true_positives
        # For whichever class the denominator is zero, we output a F1 score for this class of zero
        f1_scores = np.divide(
            numerator, denominator, out=np.zeros(numerator.shape, dtype=float), where=denominator != 0
        )

        if self.config.average == "binary":
            return f1_scores[self.config.pos_label]

        if self.config.average == "macro":
            return np.mean(f1_scores, dtype=float)

        # weighted case
        total_labels_per_class = true_positives + false_negatives
        return float(np.average(f1_scores, weights=total_labels_per_class / total_samples))

    def get_name(self) -> str:
        return f"F1-{self.config.average}"
