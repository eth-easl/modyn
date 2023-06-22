# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import json
import logging
import multiprocessing as mp
import pathlib
import platform
import tempfile
from time import sleep
from typing import Generator
from unittest.mock import MagicMock, patch

import grpc
import pytest
import torch
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    DatasetInfo,
    EvaluateModelRequest,
    JsonString,
    MetricConfiguration,
    PythonString,
)
from modyn.evaluator.internal.metrics import AbstractEvaluationMetric, Accuracy
from modyn.evaluator.internal.pytorch_evaluator import PytorchEvaluator
from modyn.evaluator.internal.utils import EvaluationInfo, EvaluatorMessages
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
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


def get_mock_evaluation_transformer():
    return (
        "import torch\n"
        "def evaluation_transformer_function(label: torch.Tensor, model_output: torch.Tensor)"
        "-> tuple[torch.Tensor, torch.Tensor]:\n"
        "\treturn label, model_output"
    )


def get_mock_roc_auc_transformer():
    return (
        "import torch\n"
        "def evaluation_transformer_function(label: torch.Tensor, model_output: torch.Tensor) "
        "-> tuple[torch.Tensor, torch.Tensor]:\n"
        "\tmodel_output = label / 100.0\n"
        "\tlabel = torch.ones_like(label)\n"
        "\tlabel[0:50] = torch.zeros(50)\n"
        "\treturn label, model_output"
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
    # pylint: disable=abstract-method

    def __init__(self, dataset_id, bytes_parser, transform_list, storage_address, evaluation_id):
        self.dataset = iter([(key, key, key * 2) for key in range(100)])

    def __iter__(self) -> Generator:
        yield from self.dataset


@patch("modyn.evaluator.internal.utils.evaluation_info.dynamic_module_import")
def get_evaluation_info(
    evaluation_id: int,
    storage_address: str,
    metrics: [AbstractEvaluationMetric],
    trained_model_path: pathlib.Path,
    label_transformer: bool,
    model_dynamic_module_patch: MagicMock,
):
    model_dynamic_module_patch.return_value = MockModule()
    request = EvaluateModelRequest(
        trained_model_id=1,
        dataset_info=DatasetInfo(dataset_id="MNIST", num_dataloaders=1),
        device="cpu",
        amp=False,
        batch_size=4,
        metrics=[
            MetricConfiguration(
                name="Accuracy",
                config=JsonString(value=json.dumps({})),
                evaluation_transform_function=PythonString(value=get_mock_evaluation_transformer()),
            )
        ],
        model_id="model",
        model_configuration=JsonString(value=json.dumps({})),
        transform_list=[],
        bytes_parser=PythonString(value=get_mock_bytes_parser()),
        label_transformer=PythonString(value=get_mock_label_transformer() if label_transformer else ""),
    )

    return EvaluationInfo(request, evaluation_id, storage_address, metrics, trained_model_path)


@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def get_mock_evaluator(
    query_queue: mp.Queue,
    response_queue: mp.Queue,
    metric_result_queue: mp.Queue,
    trained_model_path: pathlib.Path,
    label_transformer: bool,
    test_insecure_channel: MagicMock,
    test_grpc_connection_established: MagicMock,
):
    evaluation_info = get_evaluation_info(
        1,
        "storage:5000",
        [Accuracy(config={}, evaluation_transform_func=get_mock_evaluation_transformer())],
        trained_model_path,
        label_transformer,
    )
    evaluator = PytorchEvaluator(
        evaluation_info, query_queue, response_queue, metric_result_queue, logging.getLogger(__name__)
    )
    return evaluator


@patch.object(PytorchEvaluator, "_load_state")
def test_evaluator_init(load_state_mock: MagicMock):
    evaluator: PytorchEvaluator = get_mock_evaluator(mp.Queue(), mp.Queue(), mp.Queue(), "trained_model.modyn", True)

    assert isinstance(evaluator._model, MockModelWrapper)
    assert isinstance(evaluator._model.model, MockModel)
    assert len(evaluator._metrics) == 1
    assert isinstance(evaluator._metrics[0], Accuracy)
    assert evaluator._dataloader is not None
    assert torch.all(
        torch.eq(
            torch.cat(evaluator._metrics[0].evaluation_transformer_function(torch.ones(5), torch.ones(5))),
            torch.ones(10),
        )
    )
    assert torch.all(torch.eq(evaluator._label_tranformer_function(torch.ones(5) * 2) + 0.5, torch.ones(5) * 2 + 0.5))
    assert evaluator._device == "cpu"
    assert evaluator._device_type == "cpu"
    assert not evaluator._amp
    assert evaluator._num_samples == 0
    load_state_mock.assert_called_once_with("trained_model.modyn")


@patch.object(PytorchEvaluator, "_load_state")
def test_no_transform_evaluator_init(load_state_mock: MagicMock):
    evaluator: PytorchEvaluator = get_mock_evaluator(mp.Queue(), mp.Queue(), mp.Queue(), "trained_model.modyn", False)

    assert isinstance(evaluator._model, MockModelWrapper)
    assert isinstance(evaluator._model.model, MockModel)
    assert len(evaluator._metrics) == 1
    assert isinstance(evaluator._metrics[0], Accuracy)
    assert evaluator._dataloader is not None
    assert not evaluator._label_tranformer_function
    assert evaluator._device == "cpu"
    assert evaluator._device_type == "cpu"
    assert not evaluator._amp
    assert evaluator._num_samples == 0
    load_state_mock.assert_called_once_with("trained_model.modyn")


def test_load_model():
    with tempfile.TemporaryDirectory() as temp:
        model_path = pathlib.Path(temp) / "model.modyn"
        model = MockModel()
        dict_to_save = {"model": model.state_dict()}

        torch.save(dict_to_save, model_path)

        evaluator: PytorchEvaluator = get_mock_evaluator(mp.Queue(), mp.Queue(), mp.Queue(), model_path, False)

        assert evaluator._model.model.state_dict() == dict_to_save["model"]
        assert torch.all(torch.eq(evaluator._model.model(torch.ones(4)), torch.ones(4) * 2))
        assert not model_path.is_file()


@patch.object(PytorchEvaluator, "_load_state")
def test_send_status_to_server(load_state_mock: MagicMock):
    response_queue = mp.Queue()
    evaluator: PytorchEvaluator = get_mock_evaluator(
        mp.Queue(), response_queue, mp.Queue(), "trained_model.modyn", True
    )

    evaluator.send_status_to_server(20)
    response = response_queue.get()
    assert response["num_batches"] == 20
    assert response["num_samples"] == 0


@patch("modyn.evaluator.internal.pytorch_evaluator.EvaluationDataset", MockEvaluationDataset)
@patch.object(PytorchEvaluator, "_load_state")
def test_train_invalid_query_message(load_state_mock: MagicMock):
    query_status_queue = mp.Queue()
    response_queue = mp.Queue()
    evaluator: PytorchEvaluator = get_mock_evaluator(
        query_status_queue, response_queue, mp.Queue(), "trained_model.modyn", True
    )

    query_status_queue.put("INVALID MESSAGE")
    timeout = 5
    elapsed = 0
    while query_status_queue.empty():
        sleep(1)
        elapsed += 1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within time limit.")

    with pytest.raises(ValueError):
        evaluator.evaluate()

    elapsed = 0
    while not (query_status_queue.empty() and response_queue.empty()):
        sleep(1)
        elapsed += 1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within time limit.")


@patch("modyn.evaluator.internal.pytorch_evaluator.EvaluationDataset", MockEvaluationDataset)
@patch.object(PytorchEvaluator, "_load_state")
def test_train(load_state_mock: MagicMock):
    query_status_queue = mp.Queue()
    response_queue = mp.Queue()
    metric_result_queue = mp.Queue()
    evaluator: PytorchEvaluator = get_mock_evaluator(
        query_status_queue, response_queue, metric_result_queue, "trained_model.modyn", True
    )

    query_status_queue.put(EvaluatorMessages.STATUS_QUERY_MESSAGE)
    timeout = 2
    elapsed = 0

    while query_status_queue.empty():
        sleep(0.1)
        elapsed += 0.1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    evaluator.evaluate()
    assert evaluator._num_samples == 100
    elapsed = 0
    while not query_status_queue.empty():
        sleep(0.1)
        elapsed += 0.1

        if elapsed >= timeout:
            raise TimeoutError("Did not reach desired queue state within timelimit.")

    elapsed = 0
    while True:
        if not platform.system() == "Darwin":
            if response_queue.qsize() == 1 and metric_result_queue.qsize() == 1:
                break
        else:
            if not response_queue.empty() and not metric_result_queue.empty():
                break

        sleep(0.1)
        elapsed += 0.1

        if elapsed >= timeout:
            raise AssertionError("Did not reach desired queue state after 5 seconds.")

    status = response_queue.get()
    assert status["num_batches"] == 0
    assert status["num_samples"] == 0

    # accuracy metric
    metric_name, metric_result = metric_result_queue.get()
    assert metric_name == Accuracy.get_name()
    assert metric_result == pytest.approx(1)
