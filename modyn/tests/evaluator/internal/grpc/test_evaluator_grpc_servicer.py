# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
# ruff: noqa: N802  # grpc functions are not snake case

import json
import multiprocessing as mp
import os
import pathlib
import platform
import tempfile
from time import sleep
from unittest import mock
from unittest.mock import MagicMock, call, patch

import pytest
from modyn.config.schema.pipeline import AccuracyMetricConfig, F1ScoreMetricConfig
from modyn.evaluator.internal.grpc.evaluator_grpc_servicer import EvaluatorGRPCServicer
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    DatasetInfo,
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluationAbortedReason,
    EvaluationCleanupRequest,
    EvaluationInterval,
    EvaluationResultRequest,
    EvaluationStatusRequest,
)
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import JsonString as EvaluatorJsonString
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import PythonString
from modyn.evaluator.internal.metrics import Accuracy, F1Score
from modyn.evaluator.internal.utils import EvaluationInfo, EvaluationProcessInfo
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.storage.internal.grpc.generated.storage_pb2 import GetDatasetSizeRequest, GetDatasetSizeResponse
from pydantic import ValidationError

DATABASE = pathlib.Path(os.path.abspath(__file__)).parent / "test_evaluator.database"


def get_modyn_config():
    return {
        "evaluator": {"hostname": "localhost", "port": "50000"},
        "model_storage": {"hostname": "localhost", "port": "50051", "ftp_port": "5223"},
        "storage": {"hostname": "storage", "port": "50052"},
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "hostname": "",
            "port": 0,
            "database": f"{DATABASE}",
        },
    }


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    DATABASE.unlink(True)

    with MetadataDatabaseConnection(get_modyn_config()) as database:
        database.create_tables()

        database.register_pipeline(
            1,
            "ResNet18",
            json.dumps({}),
            True,
            "{}",
            "{}",
            ModelStorageStrategyConfig(name="PyTorchFullModel"),
            incremental_model_strategy=None,
            full_model_interval=None,
        )
        database.add_trained_model(1, 10, "trained_model.modyn", "trained_model.metadata")
        database.add_trained_model(1, 11, "trained_model2.modyn", "trained_model.metadata")

    yield
    DATABASE.unlink()


class DummyModelWrapper:
    def __init__(self, model_configuration=None) -> None:
        self.model = None


class DummyModelStorageStub:
    # pylint: disable-next=invalid-name
    def FetchModel(self, request: FetchModelRequest) -> FetchModelResponse:
        if request.model_id == 1:
            return FetchModelResponse(success=True, model_path="trained_model.modyn", checksum=bytes(5))
        return FetchModelResponse(success=False)


class DummyStorageStub:
    # pylint: disable-next=invalid-name
    def GetDatasetSize(self, request: GetDatasetSizeRequest) -> GetDatasetSizeResponse:
        if request.dataset_id == "MNIST":
            return GetDatasetSizeResponse(success=True, num_keys=1000)
        return GetDatasetSizeResponse(success=False)


def get_evaluation_process_info():
    exception_queue = mp.Queue()
    metric_result_queue = mp.Queue()

    evaluation_process_info = EvaluationProcessInfo(mp.Process(), exception_queue, metric_result_queue)
    return evaluation_process_info


def get_mock_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')"


def get_mock_label_transformer():
    return (
        "import torch\ndef label_transformer_function(x: torch.Tensor) -> "
        "torch.Tensor:\n\treturn x.to(torch.float32)"
    )


def get_mock_evaluation_transformer():
    return (
        "import torch\ndef evaluation_transformer(model_output: torch.Tensor) -> "
        "torch.Tensor:\n\treturn torch.abs(model_output)"
    )


def get_evaluate_model_request(intervals=[(None, None)]):
    return EvaluateModelRequest(
        model_id=1,
        dataset_info=DatasetInfo(
            dataset_id="MNIST",
            num_dataloaders=1,
            evaluation_intervals=[
                EvaluationInterval(
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                )
                for start_timestamp, end_timestamp in intervals
            ],
        ),
        device="cpu",
        batch_size=4,
        metrics=[EvaluatorJsonString(value=AccuracyMetricConfig().model_dump_json())],
        transform_list=[],
        bytes_parser=PythonString(value=get_mock_bytes_parser()),
        label_transformer=PythonString(value=""),
    )


def get_evaluation_info(evaluation_id, model_path: pathlib.Path, config: dict, intervals=[(None, None)]):
    storage_address = f"{config['storage']['hostname']}:{config['storage']['port']}"
    return EvaluationInfo(
        request=get_evaluate_model_request(intervals),
        evaluation_id=evaluation_id,
        model_class_name="ResNet18",
        amp=False,
        model_config="{}",
        storage_address=storage_address,
        metrics=[Accuracy(AccuracyMetricConfig()), F1Score(F1ScoreMetricConfig(num_classes=2, average="macro"))],
        model_path=model_path,
    )


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_init(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator_server = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        assert evaluator_server._model_storage_stub is not None
        assert evaluator_server._storage_stub is not None
        assert evaluator_server._storage_address == "storage:50052"
        test_connect_to_model_storage.assert_called_with("localhost:50051")
        test_connect_to_storage.assert_called_with("storage:50052")


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch("modyn.evaluator.internal.grpc.evaluator_grpc_servicer.hasattr", return_value=False)
def test_evaluate_model_dynamic_module_import(
    test_has_attribute, test_connect_to_model_storage, test_connect_to_storage
):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        response = evaluator.evaluate_model(get_evaluate_model_request(), None)
        assert not response.evaluation_started
        assert not evaluator._evaluation_dict
        assert evaluator._next_evaluation_id == 0
        assert response.eval_aborted_reason == EvaluationAbortedReason.MODEL_IMPORT_FAILURE


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_evaluate_model_invalid(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        req = get_evaluate_model_request()
        # only model_id 1 and 2 exist in metadatabase, see setup_and_teardown
        req.model_id = 15
        resp = evaluator.evaluate_model(req, None)
        assert not resp.evaluation_started
        assert resp.eval_aborted_reason == EvaluationAbortedReason.MODEL_NOT_EXIST_IN_METADATA

        req = get_evaluate_model_request()
        req.dataset_info.dataset_id = "unknown"
        resp = evaluator.evaluate_model(req, None)
        assert not resp.evaluation_started
        assert evaluator._next_evaluation_id == 0
        assert resp.eval_aborted_reason == EvaluationAbortedReason.DATASET_NOT_FOUND

        req = get_evaluate_model_request()
        req.model_id = 2
        resp = evaluator.evaluate_model(req, None)
        assert not resp.evaluation_started
        assert resp.eval_aborted_reason == EvaluationAbortedReason.MODEL_NOT_EXIST_IN_STORAGE


@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_evaluate_model_empty_dataset(test_connect_to_model_storage):
    storage_stub_mock = mock.Mock(spec=["GetDatasetSize"])
    storage_stub_mock.GetDatasetSize.return_value = GetDatasetSizeResponse(success=True, num_keys=0)
    with patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=storage_stub_mock):
        with tempfile.TemporaryDirectory() as modyn_temp:
            evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
            req = get_evaluate_model_request()
            resp = evaluator.evaluate_model(req, None)
            assert not resp.evaluation_started
            assert evaluator._next_evaluation_id == 0
            assert resp.eval_aborted_reason == EvaluationAbortedReason.EMPTY_DATASET


@patch("modyn.evaluator.internal.grpc.evaluator_grpc_servicer.download_trained_model", return_value=None)
@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_evaluate_model_download_trained_model(
    test_connect_to_model_storage, test_connect_to_storage, test_download_trained_model
):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        resp = evaluator.evaluate_model(get_evaluate_model_request(), None)
        assert not resp.evaluation_started
        assert resp.eval_aborted_reason == EvaluationAbortedReason.DOWNLOAD_MODEL_FAILURE


@patch.object(EvaluatorGRPCServicer, "_run_evaluation")
@patch(
    "modyn.evaluator.internal.grpc.evaluator_grpc_servicer.download_trained_model",
    return_value=pathlib.Path("downloaded_model.modyn"),
)
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_evaluate_model_correct_time_range_used_and_correct_data_sizes_returned(
    test_connect_to_model_storage, test_download_trained_model, test__run_evaluation
) -> None:
    intervals = [(None, None), (1000, None), (None, 2000), (1000, 2000)]
    expected_dataset_sizes = [400, 200, 300, 100]

    def fake_get_dataset_size(request: GetDatasetSizeRequest):
        if request.HasField("start_timestamp") and request.HasField("end_timestamp"):
            resp = GetDatasetSizeResponse(success=True, num_keys=100)
        elif request.HasField("start_timestamp") and not request.HasField("end_timestamp"):
            resp = GetDatasetSizeResponse(success=True, num_keys=200)
        elif not request.HasField("start_timestamp") and request.HasField("end_timestamp"):
            resp = GetDatasetSizeResponse(success=True, num_keys=300)
        else:
            resp = GetDatasetSizeResponse(success=True, num_keys=400)
        return resp

    storage_stub_mock = mock.Mock(spec=["GetDatasetSize"])
    storage_stub_mock.GetDatasetSize.side_effect = fake_get_dataset_size

    with patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=storage_stub_mock):
        with tempfile.TemporaryDirectory() as modyn_temp:

            evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
            req = get_evaluate_model_request(intervals)
            resp = evaluator.evaluate_model(req, None)
            assert resp.evaluation_started
            assert resp.dataset_sizes == expected_dataset_sizes

            storage_stub_mock.GetDatasetSize.assert_has_calls(
                [
                    call(GetDatasetSizeRequest(dataset_id="MNIST", start_timestamp=None, end_timestamp=None)),
                    call(GetDatasetSizeRequest(dataset_id="MNIST", start_timestamp=1000, end_timestamp=None)),
                    call(GetDatasetSizeRequest(dataset_id="MNIST", start_timestamp=None, end_timestamp=2000)),
                    call(GetDatasetSizeRequest(dataset_id="MNIST", start_timestamp=1000, end_timestamp=2000)),
                ]
            )


@patch(
    "modyn.evaluator.internal.grpc.evaluator_grpc_servicer.download_trained_model",
    return_value=pathlib.Path("downloaded_model.modyn"),
)
@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_evaluate_model_valid(test_connect_to_model_storage, test_connect_to_storage, download_model_mock) -> None:
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        resp: EvaluateModelResponse = evaluator.evaluate_model(get_evaluate_model_request(), None)
        assert 0 in evaluator._evaluation_process_dict
        assert evaluator._next_evaluation_id == 1
        download_model_mock.assert_called_once()
        kwargs = download_model_mock.call_args.kwargs
        remote_file_path = kwargs["remote_path"]
        base_directory = kwargs["base_directory"]
        identifier = kwargs["identifier"]

        assert str(remote_file_path) == "trained_model.modyn"
        assert base_directory == evaluator._base_dir
        assert identifier == 0
        assert resp.evaluation_started
        assert resp.evaluation_id == identifier
        assert str(evaluator._evaluation_dict[resp.evaluation_id].model_path) == "downloaded_model.modyn"


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_setup_metrics(test_connect_to_model_storage, test_connect_to_storage) -> None:
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))

        acc_metric_config = AccuracyMetricConfig().model_dump_json()
        metrics = evaluator._setup_metrics([acc_metric_config])

        assert len(metrics) == 1
        assert isinstance(metrics[0], Accuracy)

        unknown_metric_config = '{"name": "UnknownMetric", "config": "", "evaluation_transformer_function": ""}'
        with pytest.raises(ValidationError):
            evaluator._setup_metrics([unknown_metric_config])

        metrics = evaluator._setup_metrics([acc_metric_config, acc_metric_config])
        assert len(metrics) == 1
        assert isinstance(metrics[0], Accuracy)


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_setup_metrics_multiple_f1(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))

        macro_f1_config = F1ScoreMetricConfig(
            evaluation_transformer_function="",
            num_classes=2,
            average="macro",
        ).model_dump_json()

        micro_f1_config = F1ScoreMetricConfig(
            evaluation_transformer_function="",
            num_classes=2,
            average="micro",
        ).model_dump_json()

        # not double macro, but macro and micro work
        metrics = evaluator._setup_metrics([macro_f1_config, micro_f1_config, macro_f1_config])

        assert len(metrics) == 2
        assert isinstance(metrics[0], F1Score)
        assert isinstance(metrics[1], F1Score)
        assert metrics[0].config.average == "macro"
        assert metrics[1].config.average == "micro"
        assert metrics[0].get_name() == "F1-macro"
        assert metrics[1].get_name() == "F1-micro"


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_get_evaluation_status_not_registered(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        response = evaluator.get_evaluation_status(EvaluationStatusRequest(evaluation_id=1), None)
        assert not response.valid


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(EvaluatorGRPCServicer, "_check_for_evaluation_exception")
def test_get_evaluation_status_alive(
    test_check_for_evaluation_exception,
    test_is_alive,
    test_connect_to_model_storage,
    test_connect_to_storage,
):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluator._evaluation_process_dict[0] = get_evaluation_process_info()
        evaluator._evaluation_dict[0] = None

        response = evaluator.get_evaluation_status(EvaluationStatusRequest(evaluation_id=0), None)
        assert response.valid
        assert response.is_running
        test_check_for_evaluation_exception.assert_not_called()


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch.object(mp.Process, "is_alive", return_value=False)
@patch.object(EvaluatorGRPCServicer, "_check_for_evaluation_exception")
@pytest.mark.parametrize("exception", [None, "exception"])
def test_get_evaluation_status_finished(
    test_check_for_evaluation_exception,
    test_is_alive,
    test_connect_to_model_storage,
    test_connect_to_storage,
    exception: str,
):
    test_check_for_evaluation_exception.return_value = exception
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluator._evaluation_process_dict[2] = get_evaluation_process_info()
        evaluator._evaluation_dict[2] = None

        response = evaluator.get_evaluation_status(EvaluationStatusRequest(evaluation_id=2), None)
        assert response.valid
        assert not response.is_running
        if exception is None:
            assert not response.HasField("exception")
        else:
            assert response.HasField("exception")
            assert response.exception == exception


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_check_for_evaluation_exception_not_found(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluator._evaluation_process_dict[0] = get_evaluation_process_info()
        child_exception = evaluator._check_for_evaluation_exception(0)
        assert child_exception is None


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_check_for_evaluation_exception_found(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluation_process_info = get_evaluation_process_info()
        evaluator._evaluation_process_dict[1] = evaluation_process_info

        exception_msg = "big_exception"
        evaluation_process_info.exception_queue.put(exception_msg)

        child_exception = evaluator._check_for_evaluation_exception(1)
        assert child_exception == exception_msg


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_get_evaluation_result_model_not_registered(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        response = evaluator.get_evaluation_result(EvaluationResultRequest(evaluation_id=0), None)
        assert not response.valid


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch.object(mp.Process, "is_alive", return_value=True)
def test_get_evaluation_result_still_running(test_is_alive, test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluator._evaluation_process_dict[5] = get_evaluation_process_info()
        response = evaluator.get_evaluation_result(EvaluationResultRequest(evaluation_id=5), None)
        assert not response.valid


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch.object(mp.Process, "is_alive", return_value=False)
def test_get_evaluation_result_missing_metric(test_is_alive, test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluation_process_info = get_evaluation_process_info()
        evaluator._evaluation_process_dict[3] = evaluation_process_info
        config = get_modyn_config()
        evaluator._evaluation_dict[3] = get_evaluation_info(3, pathlib.Path("trained.model"), config)
        response = evaluator.get_evaluation_result(EvaluationResultRequest(evaluation_id=3), None)
        assert not response.valid


@patch.object(Accuracy, "get_evaluation_result", side_effect=[0.5, 0.72, 0.3])
@patch.object(F1Score, "get_evaluation_result", side_effect=[0.6, 0.75, 0.4])
@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch.object(mp.Process, "is_alive", return_value=False)
def test_get_evaluation_result(
    test_is_alive,
    test_connect_to_model_storage,
    test_connect_to_storage,
    test_f1: MagicMock,
    test_acc: MagicMock,
):
    intervals = [(4, None), (None, 8), (4, 8)]
    with tempfile.TemporaryDirectory() as temp:
        config = get_modyn_config()
        evaluator = EvaluatorGRPCServicer(config, pathlib.Path(temp))
        evaluator._evaluation_dict[1] = get_evaluation_info(
            1, pathlib.Path(temp) / "trained_model.modyn", config, intervals=intervals
        )

        assert len(evaluator._evaluation_dict[1].metrics) == 2
        assert isinstance(evaluator._evaluation_dict[1].metrics[0], Accuracy)
        assert isinstance(evaluator._evaluation_dict[1].metrics[1], F1Score)

        evaluation_process_info = get_evaluation_process_info()
        evaluator._evaluation_process_dict[1] = evaluation_process_info
        for _ in intervals:
            metric_res = []
            for metric in evaluator._evaluation_dict[1].metrics:
                metric_res.append((metric.get_name(), metric.get_evaluation_result()))
            evaluation_process_info.metric_result_queue.put(metric_res)

        timeout = 5
        elapsed = 0

        while True:
            if not platform.system() == "Darwin":
                if evaluation_process_info.metric_result_queue.qsize() == len(intervals):
                    break
            else:
                if not evaluation_process_info.metric_result_queue.empty():
                    break

            sleep(0.1)
            elapsed += 0.1

            if elapsed >= timeout:
                raise AssertionError("Did not reach desired queue state after 5 seconds.")

        response = evaluator.get_evaluation_result(EvaluationResultRequest(evaluation_id=1), None)
        assert response.valid
        assert len(response.evaluation_results) == len(intervals)
        expected_metric_results = [(0.5, 0.6), (0.72, 0.75), (0.3, 0.4)]
        for single_eval_data, expected_single_metric_results in zip(
            response.evaluation_results, expected_metric_results
        ):
            assert len(single_eval_data.evaluation_data) == 2
            assert single_eval_data.evaluation_data[0].metric == Accuracy(AccuracyMetricConfig()).get_name()
            assert (
                single_eval_data.evaluation_data[1].metric
                == F1Score(F1ScoreMetricConfig(num_classes=2, average="macro")).get_name()
            )
            assert single_eval_data.evaluation_data[0].result == pytest.approx(expected_single_metric_results[0])
            assert single_eval_data.evaluation_data[1].result == pytest.approx(expected_single_metric_results[1])

        assert test_acc.call_count == 3
        assert test_f1.call_count == 3


@patch.object(mp.Process, "is_alive", side_effect=[False, True, False, True, True])
@patch.object(mp.Process, "terminate")
@patch.object(mp.Process, "join")
@patch.object(mp.Process, "kill")
@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_cleanup_evaluations(
    test_connect_to_model_storage: MagicMock,
    test_connect_to_storage: MagicMock,
    test_kill: MagicMock,
    test_join: MagicMock,
    test_terminate: MagicMock,
    test_is_alive: MagicMock,
) -> None:
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluation_process_info = get_evaluation_process_info()
        evaluator._evaluation_process_dict[2] = evaluation_process_info
        evaluator._evaluation_process_dict[3] = evaluation_process_info
        evaluator._evaluation_process_dict[5] = evaluation_process_info
        evaluator._evaluation_dict[1] = None
        evaluator._evaluation_dict[2] = None
        evaluator._evaluation_dict[3] = None
        evaluator._evaluation_dict[5] = None
        response = evaluator.cleanup_evaluations(EvaluationCleanupRequest(evaluation_ids=[1, 2, 3, 5]), None)
        assert response.succeeded == [
            1,  # already clean
            2,  # not clean, process is dead
            3,  # not clean, but process is still alive, terminate worked
            5,  # not clean, process is still alive, terminate failed
        ]

    assert len(evaluator._evaluation_dict.keys()) == 0
    test_kill.assert_called_once()
    assert test_join.call_count == 2
    assert test_terminate.call_count == 2
    assert test_is_alive.call_count == 5
