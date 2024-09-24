# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
# ruff: noqa: N802  # grpc functions are not snake case

import json
import multiprocessing as mp
import os
import pathlib
import platform
import tempfile
import threading
from collections import defaultdict
from time import sleep
from unittest import mock
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from modyn.config.schema.pipeline import AccuracyMetricConfig
from modyn.evaluator.internal.grpc.evaluator_grpc_servicer import EvaluatorGRPCServicer
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    DatasetInfo,
    EvaluateModelIntervalResponse,
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluationAbortedReason,
    EvaluationCleanupRequest,
    EvaluationInterval,
    EvaluationResultRequest,
    EvaluationStatusRequest,
    PythonString,
)
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import JsonString as EvaluatorJsonString
from modyn.evaluator.internal.utils import EvaluationInfo, EvaluationProcessInfo
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.storage.internal.grpc.generated.storage_pb2 import GetDatasetSizeRequest, GetDatasetSizeResponse

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


def get_evaluate_model_request(intervals=None):
    if intervals is None:
        intervals = [(None, None)]
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


def get_evaluation_info(evaluation_id, model_path: pathlib.Path, config: dict, intervals=None):
    if intervals is None:
        intervals = [(None, None)]
    storage_address = f"{config['storage']['hostname']}:{config['storage']['port']}"
    return EvaluationInfo(
        request=get_evaluate_model_request(intervals),
        evaluation_id=evaluation_id,
        model_class_name="ResNet18",
        amp=True,
        model_config="{}",
        storage_address=storage_address,
        model_path=model_path,
        not_failed_interval_ids=list(range(len(intervals))),
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
        assert response.interval_responses == [
            EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.MODEL_IMPORT_FAILURE)
        ]


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
        assert resp.interval_responses == [
            EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.MODEL_NOT_EXIST_IN_METADATA)
        ]

        req = get_evaluate_model_request()
        req.dataset_info.dataset_id = "unknown"
        resp = evaluator.evaluate_model(req, None)
        assert not resp.evaluation_started
        assert evaluator._next_evaluation_id == 0
        assert resp.interval_responses == [
            EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.DATASET_NOT_FOUND)
        ]

        req = get_evaluate_model_request()
        req.model_id = 2
        resp = evaluator.evaluate_model(req, None)
        assert not resp.evaluation_started
        assert resp.interval_responses == [
            EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.MODEL_NOT_EXIST_IN_STORAGE)
        ]


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
            assert resp.interval_responses == [
                EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.EMPTY_DATASET)
            ]


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
        assert resp.interval_responses == [
            EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.DOWNLOAD_MODEL_FAILURE)
        ]


@patch.object(EvaluatorGRPCServicer, "_run_evaluation")
@patch(
    "modyn.evaluator.internal.grpc.evaluator_grpc_servicer.download_trained_model",
    return_value=pathlib.Path("downloaded_model.modyn"),
)
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch("modyn.evaluator.internal.grpc.evaluator_grpc_servicer.EvaluationInfo", wraps=EvaluationInfo)
def test_evaluate_model_mixed(
    test_evaluation_info, test_connect_to_model_storage, test_download_trained_model, test__run_evaluation
) -> None:
    intervals = [(None, None), (200, 300), (1000, None), (2000, 1000), (None, 2000), (1000, 2000)]
    expected_interval_responses = [
        EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.NOT_ABORTED, dataset_size=400),
        EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.DATASET_NOT_FOUND),
        EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.NOT_ABORTED, dataset_size=200),
        EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.EMPTY_DATASET),
        EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.NOT_ABORTED, dataset_size=300),
        EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.NOT_ABORTED, dataset_size=100),
    ]

    def fake_get_dataset_size(request: GetDatasetSizeRequest):
        if request.HasField("start_timestamp") and request.HasField("end_timestamp"):
            if request.start_timestamp == 1000 and request.end_timestamp == 2000:
                resp = GetDatasetSizeResponse(success=True, num_keys=100)
            elif request.start_timestamp == 2000 and request.end_timestamp == 1000:
                resp = GetDatasetSizeResponse(success=True, num_keys=0)
            else:
                resp = GetDatasetSizeResponse(success=False)
        elif request.HasField("start_timestamp") and not request.HasField("end_timestamp"):
            resp = GetDatasetSizeResponse(success=True, num_keys=200)
        elif not request.HasField("start_timestamp") and request.HasField("end_timestamp"):
            resp = GetDatasetSizeResponse(success=True, num_keys=300)
        else:
            resp = GetDatasetSizeResponse(success=True, num_keys=400)
        return resp

    storage_stub_mock = mock.Mock(spec=["GetDatasetSize"])
    storage_stub_mock.GetDatasetSize.side_effect = fake_get_dataset_size

    req = get_evaluate_model_request(intervals)
    with patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=storage_stub_mock):
        with tempfile.TemporaryDirectory() as modyn_temp:
            evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))

            resp = evaluator.evaluate_model(req, None)
            assert resp.evaluation_started
            assert len(resp.interval_responses) == len(intervals)
            assert resp.interval_responses == expected_interval_responses

            storage_stub_mock.GetDatasetSize.assert_has_calls(
                [
                    call(GetDatasetSizeRequest(dataset_id="MNIST", start_timestamp=None, end_timestamp=None)),
                    call(GetDatasetSizeRequest(dataset_id="MNIST", start_timestamp=200, end_timestamp=300)),
                    call(GetDatasetSizeRequest(dataset_id="MNIST", start_timestamp=1000, end_timestamp=None)),
                    call(GetDatasetSizeRequest(dataset_id="MNIST", start_timestamp=2000, end_timestamp=1000)),
                    call(GetDatasetSizeRequest(dataset_id="MNIST", start_timestamp=None, end_timestamp=2000)),
                    call(GetDatasetSizeRequest(dataset_id="MNIST", start_timestamp=1000, end_timestamp=2000)),
                ]
            )
    test_evaluation_info.assert_called_with(req, 0, "ResNet18", "{}", True, ANY, ANY, [0, 2, 4, 5])


@patch(
    "modyn.evaluator.internal.grpc.evaluator_grpc_servicer.download_trained_model",
    return_value=pathlib.Path("downloaded_model.modyn"),
)
@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_evaluate_model_valid(test_connect_to_model_storage, test_connect_to_storage, download_model_mock) -> None:
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        all_eval_dicts = [
            evaluator._evaluation_process_dict,
            evaluator._evaluation_dict,
            evaluator._evaluation_data_dict,
            evaluator._evaluation_data_dict_locks,
        ]
        assert all([0 not in eval_dict for eval_dict in all_eval_dicts])
        resp: EvaluateModelResponse = evaluator.evaluate_model(get_evaluate_model_request(), None)
        assert all([0 in eval_dict for eval_dict in all_eval_dicts])
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
def test_get_evaluation_status_not_registered(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        response = evaluator.get_evaluation_status(EvaluationStatusRequest(evaluation_id=1), None)
        assert not response.valid


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(EvaluatorGRPCServicer, "_check_for_evaluation_exception")
@patch.object(EvaluatorGRPCServicer, "_drain_result_queue")
def test_get_evaluation_status_alive(
    test__drain_result_queue,
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
@patch.object(EvaluatorGRPCServicer, "_drain_result_queue")
@pytest.mark.parametrize("exception", [None, "exception"])
def test_get_evaluation_status_finished(
    test__drain_result_queue,
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


def fake_evaluate(
    evaluation_info: EvaluationInfo, log_path: pathlib.Path, exception_queue: mp.Queue, metric_result_queue: mp.Queue
):
    num_evals = len(evaluation_info.not_failed_interval_ids)
    for idx_idx, interval_idx in enumerate(evaluation_info.not_failed_interval_ids):
        if idx_idx == num_evals - 1:
            exception_queue.put("A fake exception for dataloader!")
        else:
            metric_result_queue.put((interval_idx, [("Accuracy", 0.5), ("F1Score", 0.6)]))


def fake_process_start(self):
    # let the target function execute directly instead of starting a new process
    self._target(*self._args, **self._kwargs)


@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(mp.Process, "start", fake_process_start)
@patch("modyn.evaluator.internal.grpc.evaluator_grpc_servicer.evaluate", fake_evaluate)
def test__run_evaluation_retain_metrics_before_real_exception(test_connect_to_storage, test_connect_to_model_storage):
    evaluation_id = 0
    modyn_config = get_modyn_config()
    intervals = [(None, 100), (100, 200)]
    exception_msg = "A fake exception for dataloader!"
    evaluation_info = get_evaluation_info(evaluation_id, pathlib.Path("trained.model"), modyn_config, intervals)
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluator._evaluation_dict[evaluation_id] = evaluation_info
        evaluator._evaluation_data_dict_locks[evaluation_id] = threading.Lock()
        evaluator._evaluation_data_dict[evaluation_id] = defaultdict(list)
        evaluator._run_evaluation(evaluation_id)

    get_status_req = EvaluationStatusRequest(evaluation_id=evaluation_id)
    get_status_resp = evaluator.get_evaluation_status(get_status_req, None)
    # since now, it's single-process execution, the evaluation is finished
    assert not get_status_resp.is_running
    assert get_status_resp.valid
    assert get_status_resp.HasField("exception")
    assert get_status_resp.exception == exception_msg

    get_result_req = EvaluationResultRequest(evaluation_id=evaluation_id)
    get_result_resp = evaluator.get_evaluation_result(get_result_req, None)
    assert not get_result_resp.valid


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
def test_get_evaluation_result_incomplete_metric(test_is_alive, test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluation_process_info = get_evaluation_process_info()
        evaluation_id = 3
        evaluator._evaluation_data_dict_locks[evaluation_id] = threading.Lock()
        evaluator._evaluation_data_dict[evaluation_id] = defaultdict(list)
        evaluator._evaluation_process_dict[evaluation_id] = evaluation_process_info
        config = get_modyn_config()
        evaluator._evaluation_dict[evaluation_id] = get_evaluation_info(
            evaluation_id, pathlib.Path("trained.model"), config, intervals=((None, 100), (100, 200))
        )
        # though we have two intervals, one metric result is available because of exception
        metric_result_queue = evaluation_process_info.metric_result_queue
        metric_result_queue.put((1, [("Accuracy", 0.5)]))
        response = evaluator.get_evaluation_result(EvaluationResultRequest(evaluation_id=3), None)
        assert not response.valid


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch.object(mp.Process, "is_alive", return_value=False)
def test_get_evaluation_result(
    test_is_alive,
    test_connect_to_model_storage,
    test_connect_to_storage,
):
    intervals = [(4, None), (None, 8), (4, 8)]
    with tempfile.TemporaryDirectory() as temp:
        config = get_modyn_config()
        evaluator = EvaluatorGRPCServicer(config, pathlib.Path(temp))
        evaluator._evaluation_dict[1] = get_evaluation_info(
            1, pathlib.Path(temp) / "trained_model.modyn", config, intervals=intervals
        )
        evaluator._evaluation_data_dict_locks[1] = threading.Lock()
        evaluator._evaluation_data_dict[1] = defaultdict(list)

        evaluation_process_info = get_evaluation_process_info()
        evaluator._evaluation_process_dict[1] = evaluation_process_info
        expected_metric_results = [(0.5, 0.6), (0.72, 0.75), (0.3, 0.4)]
        for idx, (accuracy, f1score) in enumerate(expected_metric_results):
            evaluation_process_info.metric_result_queue.put((idx, [("Accuracy", accuracy), ("F1Score", f1score)]))
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

        for expected_interval_idx, (single_eval_data, expected_single_metric_results) in enumerate(
            zip(response.evaluation_results, expected_metric_results)
        ):
            assert single_eval_data.interval_index == expected_interval_idx
            assert len(single_eval_data.evaluation_data) == 2
            assert single_eval_data.evaluation_data[0].metric == "Accuracy"
            assert single_eval_data.evaluation_data[1].metric == "F1Score"
            assert single_eval_data.evaluation_data[0].result == pytest.approx(expected_single_metric_results[0])
            assert single_eval_data.evaluation_data[1].result == pytest.approx(expected_single_metric_results[1])


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
    all_ids = [1, 2, 3, 5]
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluation_process_info = get_evaluation_process_info()
        evaluator._evaluation_process_dict[2] = evaluation_process_info
        evaluator._evaluation_process_dict[3] = evaluation_process_info
        evaluator._evaluation_process_dict[5] = evaluation_process_info
        for idx in all_ids:
            evaluator._evaluation_dict[idx] = None
            evaluator._evaluation_data_dict_locks[idx] = threading.Lock()
            evaluator._evaluation_data_dict[idx] = defaultdict(list)
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
