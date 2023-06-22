import numpy as np
import pytest
import torch
from modyn.evaluator.internal.metrics import Accuracy
from modyn.evaluator.internal.metrics.abstract_decomposable_metric import AbstractDecomposableMetric


def get_invalid_evaluation_transformer():
    return "evaluation_transformer_function=(3, 4)"


def get_mock_evaluation_transformer():
    return (
        "import torch\n"
        "def evaluation_transformer_function(label: torch.Tensor, model_output: torch.Tensor) "
        "-> tuple[torch.Tensor, torch.Tensor]:\n"
        "\ttrf_output = torch.argmax(model_output, dim=-1)\n"
        "\treturn label, trf_output"
    )


def test_accuracy_invalid_transform():
    with pytest.raises(ValueError):
        Accuracy(evaluation_transform_func=get_invalid_evaluation_transformer(), config={})


def test_accuracy_valid_transform():
    accuracy = Accuracy(evaluation_transform_func=get_mock_evaluation_transformer(), config={})

    _, trf_output = accuracy.evaluation_transformer_function(5, torch.arange(10))
    assert trf_output == 9

    y_true = torch.from_numpy(np.array([0, 0, 0, 9, 9, 9]))
    y_pred = torch.stack([torch.arange(10)] * 6)

    accuracy.batch_evaluated_callback(y_true, y_pred, 6)

    assert accuracy.get_evaluation_result() == pytest.approx(0.5)


def test_accuracy_metric():
    accuracy = Accuracy(evaluation_transform_func="", config={})

    assert isinstance(accuracy, AbstractDecomposableMetric)

    zeroes = torch.zeros(1)
    y_pred = torch.zeros(1)

    accuracy.batch_evaluated_callback(zeroes, y_pred, 1)

    assert accuracy.get_evaluation_result() == 1

    zeroes = torch.zeros((2, 10))
    y_pred = torch.ones((2, 10))

    accuracy.batch_evaluated_callback(zeroes, y_pred, 2)

    assert accuracy.get_evaluation_result() == pytest.approx(1.0 / 3)

    y_true = torch.from_numpy(np.array([0, 1, 0, 1, 0, 1]))
    y_pred = torch.from_numpy(np.array([1, 0, 1, 0, 0, 1]))

    accuracy.batch_evaluated_callback(y_true, y_pred, 6)

    assert accuracy.get_evaluation_result() == pytest.approx(1.0 / 3)


def test_accuracy_metric_invalid():
    accuracy = Accuracy(evaluation_transform_func="", config={})

    zeroes = torch.zeros((5, 1))
    y_pred = torch.zeros((4, 1))

    with pytest.raises(TypeError):
        accuracy.batch_evaluated_callback(zeroes, y_pred, 5)
