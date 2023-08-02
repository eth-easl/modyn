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
        "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
        "\treturn torch.argmax(model_output, dim=-1)"
    )


def test_accuracy_invalid_transform():
    accuracy = Accuracy(evaluation_transform_func=get_invalid_evaluation_transformer(), config={})
    with pytest.raises(ValueError):
        accuracy.deserialize_evaluation_transformer()


def test_accuracy_valid_transform():
    accuracy = Accuracy(evaluation_transform_func=get_mock_evaluation_transformer(), config={})
    accuracy.deserialize_evaluation_transformer()

    trf_output = accuracy.evaluation_transformer_function(torch.arange(10))
    assert trf_output == 9

    y_true = torch.from_numpy(np.array([0, 0, 0, 9, 9, 9]))
    y_pred = torch.stack([torch.arange(10)] * 6)

    accuracy.evaluate_batch(y_true, y_pred, 6)

    assert accuracy.get_evaluation_result() == pytest.approx(0.5)


def test_accuracy():
    accuracy = Accuracy(evaluation_transform_func="", config={})
    accuracy.deserialize_evaluation_transformer()

    assert isinstance(accuracy, AbstractDecomposableMetric)

    zeroes = torch.zeros(3)
    y_pred = torch.zeros(3)

    accuracy.evaluate_batch(zeroes, y_pred, 3)

    assert accuracy.get_evaluation_result() == 1

    zeroes = torch.zeros(6)
    y_pred = torch.ones(6)

    accuracy.evaluate_batch(zeroes, y_pred, 6)

    assert accuracy.get_evaluation_result() == pytest.approx(1.0 / 3)

    y_true = torch.from_numpy(np.array([0, 1, 0, 1, 0, 1]))
    y_pred = torch.from_numpy(np.array([1, 0, 1, 0, 0, 1]))

    accuracy.evaluate_batch(y_true, y_pred, 6)

    assert accuracy.get_evaluation_result() == pytest.approx(1.0 / 3)


def test_accuracy_invalid():
    accuracy = Accuracy(evaluation_transform_func="", config={})
    accuracy.deserialize_evaluation_transformer()

    zeroes = torch.zeros(5)
    y_pred = torch.zeros(4)

    with pytest.raises(TypeError):
        accuracy.evaluate_batch(zeroes, y_pred, 5)
