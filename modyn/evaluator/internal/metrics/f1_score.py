from enum import Enum
from typing import Any

import numpy as np
import torch
from modyn.evaluator.internal.metrics.abstract_decomposable_metric import AbstractDecomposableMetric


class F1ScoreTypes(Enum):
    MACRO = "macro"
    MICRO = "micro"
    WEIGHTED = "weighted"
    BINARY = "binary"


VALID_SCORE_TYPES = {score_type.value for score_type in F1ScoreTypes}


class F1Score(AbstractDecomposableMetric):
    """
    F1-score implementation. Configuration options:
    - num_classes: the total number of classes.
    - (optional) average: the method used to average f1-score in the multiclass setting (default macro).
    - (optional) pos_label: the positive label used in binary classification (default 1), only its f1-score is returned.
    """

    def __init__(self, evaluation_transform_func: str, config: dict[str, Any]) -> None:
        super().__init__(evaluation_transform_func, config)

        if "num_classes" not in config:
            raise ValueError("Must provide num_classes to the F1-score metric.")
        self.num_classes = config["num_classes"]

        self.average = F1ScoreTypes.MACRO
        if "average" in config:
            average_type_name = config["average"]
            if average_type_name in VALID_SCORE_TYPES:
                self.average = F1ScoreTypes(average_type_name)
            else:
                raise ValueError(
                    f"Provided invalid average strategy {average_type_name}. "
                    f"Must be one of: {', '.join(VALID_SCORE_TYPES)}"
                )

        self.pos_label = 1
        if "pos_label" in config:
            self.pos_label = config["pos_label"]

        if self.average == F1ScoreTypes.BINARY and self.num_classes != 2:
            raise ValueError("Must only have 2 classes for binary F1-score.")

        # store true positives (tp), false positives (fp) and false negatives (fn) in matrix
        self.classification_matrix = np.zeros((3, self.num_classes))

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
        if self.average == F1ScoreTypes.MICRO:
            return sum_tp / total_samples

        false_negatives = self.classification_matrix[2]

        denominator = 2 * true_positives + false_positives + false_negatives
        numerator = 2 * true_positives
        # For whichever class the denominator is zero, we output a F1 score for this class of zero
        f1_scores = np.divide(
            numerator, denominator, out=np.zeros(numerator.shape, dtype=float), where=denominator != 0
        )

        if self.average == F1ScoreTypes.BINARY:
            return f1_scores[self.pos_label]

        if self.average == F1ScoreTypes.MACRO:
            return np.mean(f1_scores, dtype=float)

        # weighted case
        total_labels_per_class = true_positives + false_negatives
        return float(np.average(f1_scores, weights=total_labels_per_class / total_samples))

    def get_name(self) -> str:
        return f"F1-{self.average.value}"
