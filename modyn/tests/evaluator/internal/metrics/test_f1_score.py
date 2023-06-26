import numpy as np
import pytest
import torch
from modyn.evaluator.internal.metrics import AbstractDecomposableMetric, F1Score
from modyn.evaluator.internal.metrics.f1_score import F1ScoreTypes


def get_invalid_evaluation_transformer():
    return "evaluation_transformer_function=10"


def get_mock_evaluation_transformer():
    return (
        "import torch\n"
        "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
        "\treturn torch.argmax(model_output, dim=-1)"
    )


def test_f1_score_invalid_transform():
    with pytest.raises(ValueError):
        F1Score(evaluation_transform_func=get_invalid_evaluation_transformer(), config={"num_classes": 2})


def test_f1_score_invalid():
    with pytest.raises(ValueError):
        F1Score(evaluation_transform_func="", config={"average": "micro"})
    with pytest.raises(ValueError):
        F1Score(evaluation_transform_func="", config={"num_classes": 2, "average": "unknown"})
    with pytest.raises(ValueError):
        F1Score(evaluation_transform_func="", config={"num_classes": 3, "average": "binary"})

    f1_score = F1Score(evaluation_transform_func="", config={"num_classes": 2, "average": "macro"})

    assert f1_score.get_evaluation_result() == 0

    f1_score = F1Score(evaluation_transform_func="", config={"num_classes": 3, "average": "weighted"})
    y_true = torch.from_numpy(np.array([0, 0, 0, 2]))
    y_pred = torch.from_numpy(np.array([0, 0, 0, 0]))

    f1_score.evaluate_batch(y_true, y_pred, 4)
    assert f1_score.get_evaluation_result() == 0


def test_f1_score():
    y_true = torch.from_numpy(np.array([0, 2, 2, 2, 2, 0, 1, 2, 0, 2]))
    y_pred = torch.from_numpy(np.array([0, 1, 2, 2, 1, 1, 1, 0, 0, 2]))

    f1_score = F1Score(evaluation_transform_func="", config={"num_classes": 3, "average": "micro"})
    assert isinstance(f1_score, AbstractDecomposableMetric)
    assert f1_score.average == F1ScoreTypes.MICRO

    f1_score.evaluate_batch(y_true, y_pred, 10)
    assert f1_score.get_evaluation_result() == 0.6

    f1_score = F1Score(evaluation_transform_func="", config={"num_classes": 3, "average": "macro"})
    assert f1_score.average == F1ScoreTypes.MACRO

    f1_score.evaluate_batch(y_true, y_pred, 10)
    assert f1_score.get_evaluation_result() == pytest.approx(0.577, abs=0.01)

    f1_score = F1Score(evaluation_transform_func="", config={"num_classes": 3, "average": "weighted"})
    assert f1_score.average == F1ScoreTypes.WEIGHTED

    f1_score.evaluate_batch(y_true, y_pred, 10)
    assert f1_score.get_evaluation_result() == pytest.approx(0.64)

    f1_score = F1Score(evaluation_transform_func="", config={"num_classes": 2, "average": "binary", "pos_label": 0})
    assert f1_score.average == F1ScoreTypes.BINARY
    y_true = 1 - torch.from_numpy(np.array([0, 1, 0, 1, 0, 1, 0, 1]))
    y_pred = 1 - torch.from_numpy(np.array([1, 1, 1, 1, 0, 1, 0, 0]))

    f1_score.evaluate_batch(y_true, y_pred, 8)
    assert f1_score.get_evaluation_result() == pytest.approx(2.0 / 3)


def test_f1_score_transform():
    f1_score = F1Score(get_mock_evaluation_transformer(), config={"num_classes": 2})
    assert f1_score.average == F1ScoreTypes.MACRO
    assert f1_score.evaluation_transformer_function(torch.arange(10)) == 9

    y_true = torch.from_numpy(np.array([1, 0, 0, 0, 1, 1]))
    y_pred = torch.stack([torch.arange(2)] * 6)

    f1_score.evaluate_batch(y_true, y_pred, 6)

    assert f1_score.get_evaluation_result() == pytest.approx(1 / 3)
