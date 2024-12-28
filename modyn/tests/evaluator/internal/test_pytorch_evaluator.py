import logging
import multiprocessing as mp
import pathlib
import sys
import tempfile
from collections.abc import Generator
from unittest.mock import ANY, MagicMock, call, patch

import pytest
import torch
from torch.utils.data import IterableDataset

from modyn.config import F1ScoreMetricConfig
from modyn.config.schema.pipeline import AccuracyMetricConfig
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    DatasetInfo,
    EvaluateModelRequest,
    EvaluationInterval,
    PythonString,
)
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    JsonString as EvaluatorJsonString,
)
from modyn.evaluator.internal.pytorch_evaluator import PytorchEvaluator
from modyn.evaluator.internal.utils import EvaluationInfo


def noop_constructor_mock(self, channel):
    pass


def get_mock_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')"


def get_mock_label_transformer():
    return "import torch\ndef label_transformer_function(x: torch.Tensor) -> " "torch.Tensor:\n\treturn x"


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
        return torch.zeros_like(data)


class MockModelWrapper:
    def __init__(self, model_configuration=None, device="cpu", amp=False) -> None:
        self.model = MockModel()


class MockModule:
    def __init__(self) -> None:
        self.model = MockModelWrapper

    def train(self) -> None:
        pass


class MockEvaluationDataset(IterableDataset):
    def __init__(self, input_to_output_func=lambda x: 0):
        self.dataset = iter(
            [
                (
                    key,
                    torch.tensor((key,)),
                    torch.tensor((input_to_output_func(key),), dtype=torch.int),
                )
                for key in range(100)
            ]
        )

    def __iter__(self) -> Generator:
        yield from self.dataset


EVALUATION_INTERVALS = [
    EvaluationInterval(),
    EvaluationInterval(end_timestamp=100),
    EvaluationInterval(start_timestamp=100, end_timestamp=200),
    EvaluationInterval(start_timestamp=150, end_timestamp=250),
    EvaluationInterval(start_timestamp=200),
]


@patch("modyn.evaluator.internal.utils.evaluation_info.dynamic_module_import")
def get_evaluation_info(
    evaluation_id: int,
    storage_address: str,
    metrics: list[EvaluatorJsonString],
    trained_model_path: pathlib.Path,
    label_transformer: bool,
    not_failed_interval_ids: list[int],
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
        metrics=metrics,
        transform_list=[],
        bytes_parser=PythonString(value=get_mock_bytes_parser()),
        label_transformer=PythonString(value=get_mock_label_transformer() if label_transformer else ""),
    )
    return EvaluationInfo(
        request,
        evaluation_id,
        "model",
        "{}",
        False,
        storage_address,
        trained_model_path,
        not_failed_interval_ids,
    )


def get_mock_evaluator(
    trained_model_path: str,
    label_transformer: bool,
    metric_queue: mp.Queue = mp.Queue(),
    not_failed_interval_ids=None,
    metric_jsons=None,
) -> PytorchEvaluator:
    if not_failed_interval_ids is None:
        not_failed_interval_ids = [0, 1, 2, 3]
    if metric_jsons is None:
        metric_jsons = [
            AccuracyMetricConfig(evaluation_transformer_function=get_mock_accuracy_transformer()).model_dump_json()
        ]
    proto_metrics = [EvaluatorJsonString(value=metric_json) for metric_json in metric_jsons]
    evaluation_info = get_evaluation_info(
        1,
        "storage:5000",
        proto_metrics,
        pathlib.Path(trained_model_path),
        label_transformer,
        not_failed_interval_ids,
    )
    evaluator = PytorchEvaluator(evaluation_info, logging.getLogger(__name__), metric_queue)
    return evaluator


@patch.object(PytorchEvaluator, "_load_state")
def test_evaluator_init(load_state_mock: MagicMock) -> None:
    evaluator: PytorchEvaluator = get_mock_evaluator("trained_model.modyn", True)

    assert isinstance(evaluator._model, MockModelWrapper)
    assert isinstance(evaluator._model.model, MockModel)
    assert evaluator._evaluation_id == 1
    assert torch.all(
        torch.eq(
            evaluator._label_transformer_function(torch.ones(5) * 2) + 0.5,
            torch.ones(5) * 2 + 0.5,
        )
    )
    assert evaluator._device == "cpu"
    assert not evaluator._amp
    load_state_mock.assert_called_once_with(pathlib.Path("trained_model.modyn"))


@patch.object(PytorchEvaluator, "_load_state")
def test_no_transform_evaluator_init(load_state_mock: MagicMock):
    evaluator: PytorchEvaluator = get_mock_evaluator("trained_model.modyn", False)

    assert isinstance(evaluator._model, MockModelWrapper)
    assert isinstance(evaluator._model.model, MockModel)
    assert not evaluator._label_transformer_function
    assert evaluator._device == "cpu"
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
        assert not model_path.is_file()


@pytest.mark.skipif(sys.platform == "darwin", reason="Skipping test on macOS")
@patch.object(
    PytorchEvaluator,
    "_prepare_dataloader",
    side_effect=[
        # all samples are correctly labeled
        MockEvaluationDataset(),
        # only the first sample 0 is correctly labeled
        MockEvaluationDataset(lambda x: 0 if x == 0 else 1),
        # no samples are correctly labeled
        MockEvaluationDataset(lambda x: 1),
        # half of the samples are correctly labeled
        MockEvaluationDataset(lambda x: x % 2),
    ],
)
@patch.object(PytorchEvaluator, "_load_state")
def test_evaluate(_load_state_mock: MagicMock, prepare_dataloader_mock: MagicMock):
    metric_queue = mp.Queue()
    not_failed_interval_ids = [0, 1, 2, 4]
    metric_jsons = [
        AccuracyMetricConfig(evaluation_transformer_function=get_mock_accuracy_transformer()).model_dump_json(),
        F1ScoreMetricConfig(num_classes=2).model_dump_json(),
    ]
    evaluator: PytorchEvaluator = get_mock_evaluator(
        "trained_model.modyn", True, metric_queue, not_failed_interval_ids, metric_jsons
    )
    evaluator.evaluate()

    prepare_dataloader_mock.assert_has_calls(
        [
            call(ANY, None, None),
            call(ANY, None, 100),
            call(ANY, 100, 200),
            call(ANY, 200, None),
        ]
    )

    expected_accuracies = [1.0, 0.01, 0.0, 0.5]
    expected_f1scores = [0.5, ANY, 0.0, 1 / 3]
    for idx, accuracy, f1score in zip(not_failed_interval_ids, expected_accuracies, expected_f1scores):
        # the accuracies are only correctly calculated if we correctly reset the
        res = metric_queue.get()
        assert res == (
            idx,
            [
                ("Accuracy", pytest.approx(accuracy)),
                ("F1-macro", pytest.approx(f1score)),
            ],
        )
