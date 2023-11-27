
from typing import Any
from modyn.evaluator.internal.metrics.f1_score import F1Score


class WeightedF1Score(F1Score):
    """
    Temporary Hack to allow weighted F1 and Macro FW
    - num_classes: the total number of classes.
    - (optional) average: the method used to average f1-score in the multiclass setting (default macro).
    - (optional) pos_label: the positive label used in binary classification (default 1), only its f1-score is returned.
    """

    def __init__(self, evaluation_transform_func: str, config: dict[str, Any]) -> None:
        config["average"] = "weighted"
        super().__init__(evaluation_transform_func, config)

    @staticmethod
    def get_name() -> str:
        return "WeightedF1-score"
