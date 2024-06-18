import numpy as np
import pytest
import torch
from modyn.config.schema.pipeline import F1ScoreMetricConfig
from modyn.evaluator.internal.metrics import AbstractDecomposableMetric, F1Score
from pydantic import ValidationError


def get_invalid_evaluation_transformer() -> str:
    return "evaluation_transformer_function=10"


def get_mock_evaluation_transformer() -> str:
    return (
        "import torch\n"
        "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
        "\treturn torch.argmax(model_output, dim=-1)"
    )


def test_f1_score_invalid_transform() -> None:
    with pytest.raises(ValidationError):
        F1ScoreMetricConfig(evaluation_transformer_function=get_invalid_evaluation_transformer(), num_classes=2)


def test_f1_score_invalid():
    with pytest.raises(ValidationError):
        F1ScoreMetricConfig(evaluation_transformer_function="", average="micro")
    with pytest.raises(ValidationError):
        F1ScoreMetricConfig(evaluation_transformer_function="", num_classes=2, average="unknown")
    with pytest.raises(ValidationError):
        F1ScoreMetricConfig(evaluation_transformer_function="", num_classes=3, average="unknown")

    f1_score = F1Score(F1ScoreMetricConfig(num_classes=2, average="macro"))
    assert f1_score.get_evaluation_result() == 0


def test_f1_score():
    y_true = torch.from_numpy(np.array([0, 2, 2, 2, 2, 0, 1, 2, 0, 2]))
    y_pred = torch.from_numpy(np.array([0, 1, 2, 2, 1, 1, 1, 0, 0, 2]))

    f1_score = F1Score(F1ScoreMetricConfig(num_classes=3, average="micro"))
    assert isinstance(f1_score, AbstractDecomposableMetric)
    assert f1_score.config.average == "micro"

    f1_score.evaluate_batch(y_true, y_pred, 10)
    assert f1_score.get_evaluation_result() == 0.6

    f1_score = F1Score(F1ScoreMetricConfig(num_classes=3, average="macro"))
    assert f1_score.config.average == "macro"

    f1_score.evaluate_batch(y_true, y_pred, 10)
    assert f1_score.get_evaluation_result() == pytest.approx(0.577, abs=0.01)

    f1_score = F1Score(F1ScoreMetricConfig(num_classes=3, average="weighted"))
    assert f1_score.config.average == "weighted"

    f1_score.evaluate_batch(y_true, y_pred, 10)
    assert f1_score.get_evaluation_result() == pytest.approx(0.64)

    f1_score = F1Score(F1ScoreMetricConfig(num_classes=2, average="binary", pos_label=0))
    assert f1_score.config.average == "binary"
    y_true = 1 - torch.from_numpy(np.array([0, 1, 0, 1, 0, 1, 0, 1]))
    y_pred = 1 - torch.from_numpy(np.array([1, 1, 1, 1, 0, 1, 0, 0]))

    f1_score.evaluate_batch(y_true, y_pred, 8)
    assert f1_score.get_evaluation_result() == pytest.approx(2.0 / 3)


def test_f1_score_transform():
    f1_score = F1Score(
        F1ScoreMetricConfig(
            evaluation_transformer_function=get_mock_evaluation_transformer(), num_classes=2, pos_label=0
        )
    )
    f1_score.deserialize_evaluation_transformer()
    assert f1_score.config.evaluation_transformer_function_deserialized is not None

    assert f1_score.config.average == "macro"
    assert f1_score.evaluation_transformer_function(torch.arange(10)) == 9

    y_true = torch.from_numpy(np.array([1, 0, 0, 0, 1, 1]))
    y_pred = torch.stack([torch.arange(2)] * 6)

    f1_score.evaluate_batch(y_true, y_pred, 6)

    assert f1_score.get_evaluation_result() == pytest.approx(1 / 3)
