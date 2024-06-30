# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import logging
import multiprocessing as mp
import pathlib
import tempfile
from typing import Generator
from unittest.mock import ANY, MagicMock, call, patch

import pytest
import torch
from modyn.config.schema.pipeline import AccuracyMetricConfig
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import DatasetInfo, EvaluateModelRequest, EvaluationInterval
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import JsonString as EvaluatorJsonString
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import PythonString
from modyn.evaluator.internal.metrics import AbstractEvaluationMetric, Accuracy
from modyn.evaluator.internal.pytorch_evaluator import PytorchEvaluator
from modyn.evaluator.internal.utils import EvaluationInfo
from torch.utils.data import IterableDataset


def noop_constructor_mock(self, channel):
    pass


def get_mock_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')"


def get_mock_label_transformer():
    return (
        "import torch\ndef label_transformer_function(x: torch.Tensor) -> "
        "torch.Tensor:\n\treturn x.to(torch.float32)"
    )


def get_mock_accuracy_transformer():
    return (
        "import torch\n"
        "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
        "\treturn model_output"
    )


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, data):
        return data * 2


class MockModelWrapper:
    def __init__(self, model_configuration=None, device="cpu", amp=False) -> None:
        self.model = MockModel()


class MockModule:
    def __init__(self) -> None:
        self.model = MockModelWrapper

    def train(self) -> None:
        pass


class MockEvaluationDataset(IterableDataset):

    def __init__(self, input_to_output_func=lambda x: x * 2):
        self.dataset = iter(
            [(key, torch.tensor((key,)), torch.tensor((input_to_output_func(key),))) for key in range(100)]
        )

    def __iter__(self) -> Generator:
        yield from self.dataset


EVALUATION_INTERVALS = [
    EvaluationInterval(),
    EvaluationInterval(end_timestamp=100),
    EvaluationInterval(start_timestamp=100, end_timestamp=200),
    EvaluationInterval(start_timestamp=200),
]


@patch("modyn.evaluator.internal.utils.evaluation_info.dynamic_module_import")
def get_evaluation_info(
    evaluation_id: int,
    storage_address: str,
    metrics: [AbstractEvaluationMetric],
    trained_model_path: pathlib.Path,
    label_transformer: bool,
    model_dynamic_module_patch: MagicMock,
) -> EvaluationInfo:
    model_dynamic_module_patch.return_value = MockModule()
    request = EvaluateModelRequest(
        model_id=1,
        dataset_info=DatasetInfo(
            dataset_id="MNIST",
            num_dataloaders=1,
            evaluation_intervals=EVALUATION_INTERVALS,
        ),
        device="cpu",
        batch_size=4,
        metrics=[
            EvaluatorJsonString(
                value=AccuracyMetricConfig(
                    evaluation_transformer_function=get_mock_accuracy_transformer()
                ).model_dump_json()
            )
        ],
        transform_list=[],
        bytes_parser=PythonString(value=get_mock_bytes_parser()),
        label_transformer=PythonString(value=get_mock_label_transformer() if label_transformer else ""),
    )
    return EvaluationInfo(request, evaluation_id, "model", "{}", False, storage_address, metrics, trained_model_path)


def get_mock_evaluator(trained_model_path: str, label_transformer: bool) -> PytorchEvaluator:
    evaluation_info = get_evaluation_info(
        1,
        "storage:5000",
        [Accuracy(AccuracyMetricConfig(evaluation_transformer_function=get_mock_accuracy_transformer()))],
        pathlib.Path(trained_model_path),
        label_transformer,
    )
    evaluator = PytorchEvaluator(evaluation_info, logging.getLogger(__name__))
    return evaluator


@patch.object(PytorchEvaluator, "_load_state")
def test_evaluator_init(load_state_mock: MagicMock) -> None:
    evaluator: PytorchEvaluator = get_mock_evaluator("trained_model.modyn", True)

    assert isinstance(evaluator._model, MockModelWrapper)
    assert isinstance(evaluator._model.model, MockModel)
    assert len(evaluator._metrics) == 1
    assert isinstance(evaluator._metrics[0], Accuracy)
    assert torch.all(
        torch.eq(
            evaluator._metrics[0].evaluation_transformer_function(torch.ones(5)),
            torch.ones(5),
        )
    )
    assert evaluator._evaluation_id == 1
    assert torch.all(torch.eq(evaluator._label_transformer_function(torch.ones(5) * 2) + 0.5, torch.ones(5) * 2 + 0.5))
    assert evaluator._device == "cpu"
    assert evaluator._device_type == "cpu"
    assert not evaluator._amp
    assert not evaluator._contains_holistic_metric
    load_state_mock.assert_called_once_with(pathlib.Path("trained_model.modyn"))


@patch.object(PytorchEvaluator, "_load_state")
def test_no_transform_evaluator_init(load_state_mock: MagicMock):
    evaluator: PytorchEvaluator = get_mock_evaluator("trained_model.modyn", False)

    assert isinstance(evaluator._model, MockModelWrapper)
    assert isinstance(evaluator._model.model, MockModel)
    assert len(evaluator._metrics) == 1
    assert isinstance(evaluator._metrics[0], Accuracy)
    assert not evaluator._label_transformer_function
    assert evaluator._device == "cpu"
    assert evaluator._device_type == "cpu"
    assert not evaluator._amp
    load_state_mock.assert_called_once_with(pathlib.Path("trained_model.modyn"))


def test_load_model():
    with tempfile.TemporaryDirectory() as temp:
        model_path = pathlib.Path(temp) / "model.modyn"
        model = MockModel()
        dict_to_save = {"model": model.state_dict()}

        torch.save(dict_to_save, model_path)

        evaluator: PytorchEvaluator = get_mock_evaluator(str(model_path), False)

        assert evaluator._model.model.state_dict() == dict_to_save["model"]
        assert torch.all(torch.eq(evaluator._model.model(torch.ones(4)), torch.ones(4) * 2))
        assert not model_path.is_file()


@patch.object(
    PytorchEvaluator,
    "_prepare_dataloader",
    side_effect=[
        # all samples are correctly labeled
        MockEvaluationDataset(),
        # only the first sample 0 is correctly labeled
        MockEvaluationDataset(lambda x: x * 3),
        # no samples are correctly labeled
        MockEvaluationDataset(lambda x: x - 1),
        # half of the samples are correctly labeled
        MockEvaluationDataset(lambda x: x - 1 if x % 2 == 0 else x * 2),
    ],
)
@patch.object(PytorchEvaluator, "_load_state")
def test_evaluate(_load_state_mock: MagicMock, prepare_dataloader_mock: MagicMock):
    metric_result_queue = mp.Queue()
    evaluator: PytorchEvaluator = get_mock_evaluator("trained_model.modyn", True)
    evaluator.evaluate(metric_result_queue)

    prepare_dataloader_mock.assert_has_calls(
        [
            call(ANY, None, None),
            call(ANY, None, 100),
            call(ANY, 100, 200),
            call(ANY, 200, None),
        ]
    )

    expected_accuracies = [1.0, 0.01, 0.0, 0.5]
    for accuracy in expected_accuracies:
        # the accuracies are only correctly calculated if we correctly reset the state
        assert metric_result_queue.get() == [("Accuracy", pytest.approx(accuracy))]
