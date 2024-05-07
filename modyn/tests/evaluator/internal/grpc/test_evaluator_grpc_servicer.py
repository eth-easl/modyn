# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import json
import multiprocessing as mp
import os
import pathlib
import platform
import tempfile
from time import sleep
from unittest.mock import MagicMock, patch

import pytest
from modyn.evaluator.internal.grpc.evaluator_grpc_servicer import EvaluatorGRPCServicer
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    DatasetInfo,
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluationResultRequest,
    EvaluationStatusRequest,
    JsonString,
    MetricConfiguration,
    PythonString,
)
from modyn.evaluator.internal.metrics import Accuracy, F1Score
from modyn.evaluator.internal.utils import EvaluationInfo, EvaluationProcessInfo, EvaluatorMessages
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
            "host": "",
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
    status_query_queue = mp.Queue()
    status_response_queue = mp.Queue()
    exception_queue = mp.Queue()
    metric_result_queue = mp.Queue()

    evaluation_process_info = EvaluationProcessInfo(
        mp.Process(), exception_queue, status_query_queue, status_response_queue, metric_result_queue
    )
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


def get_evaluate_model_request():
    return EvaluateModelRequest(
        model_id=1,
        dataset_info=DatasetInfo(dataset_id="MNIST", num_dataloaders=1),
        device="cpu",
        batch_size=4,
        metrics=[
            MetricConfiguration(
                name="Accuracy",
                config=JsonString(value=json.dumps({})),
                evaluation_transformer=PythonString(value=""),
            )
        ],
        transform_list=[],
        bytes_parser=PythonString(value=get_mock_bytes_parser()),
        label_transformer=PythonString(value=""),
    )


def get_evaluation_info(evaluation_id, model_path: pathlib.Path, config: dict):
    storage_address = f"{config['storage']['hostname']}:{config['storage']['port']}"
    return EvaluationInfo(
        request=get_evaluate_model_request(),
        evaluation_id=evaluation_id,
        model_class_name="ResNet18",
        amp=False,
        model_config="{}",
        storage_address=storage_address,
        metrics=[Accuracy("", {}), F1Score("", {"num_classes": 2})],
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


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_evaluate_model_invalid(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        req = get_evaluate_model_request()
        req.model_id = 15
        resp = evaluator.evaluate_model(req, None)
        assert not resp.evaluation_started

        req = get_evaluate_model_request()
        req.dataset_info.dataset_id = "unknown"
        resp = evaluator.evaluate_model(req, None)
        assert not resp.evaluation_started
        assert evaluator._next_evaluation_id == 0

        req = get_evaluate_model_request()
        req.model_id = 2
        resp = evaluator.evaluate_model(req, None)
        assert not resp.evaluation_started


@patch(
    "modyn.evaluator.internal.grpc.evaluator_grpc_servicer.download_trained_model",
    return_value=pathlib.Path("downloaded_model.modyn"),
)
@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_evaluate_model_valid(test_connect_to_model_storage, test_connect_to_storage, download_model_mock):
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
def test_setup_metrics(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))

        acc_metric_config = MetricConfiguration(
            name="Accuracy",
            config=JsonString(value=json.dumps({})),
            evaluation_transformer=PythonString(value=""),
        )
        metrics = evaluator._setup_metrics([acc_metric_config])

        assert len(metrics) == 1
        assert isinstance(metrics[0], Accuracy)

        unkown_metric_config = MetricConfiguration(
            name="UnknownMetric",
            config=JsonString(value=json.dumps({})),
            evaluation_transformer=PythonString(value=""),
        )
        with pytest.raises(NotImplementedError):
            evaluator._setup_metrics([unkown_metric_config])

        metrics = evaluator._setup_metrics([acc_metric_config, acc_metric_config])
        assert len(metrics) == 1
        assert isinstance(metrics[0], Accuracy)


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
@patch.object(EvaluatorGRPCServicer, "_get_status", return_value=(3, 50))
@patch.object(EvaluatorGRPCServicer, "_check_for_evaluation_exception")
def test_get_evaluation_status_alive(
    test_check_for_evaluation_exception,
    test_get_status,
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
        assert not response.blocked
        assert not response.HasField("exception")
        assert response.state_available
        assert response.HasField("batches_seen") and response.HasField("samples_seen")
        assert response.batches_seen == 3
        assert response.samples_seen == 50
        test_check_for_evaluation_exception.assert_not_called()


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch.object(mp.Process, "is_alive", return_value=True)
@patch.object(EvaluatorGRPCServicer, "_get_status", return_value=(None, None))
@patch.object(EvaluatorGRPCServicer, "_check_for_evaluation_exception")
def test_get_evaluation_status_alive_blocked(
    test_check_for_evaluation_exception,
    test_get_status,
    test_is_alive,
    test_connect_to_model_storage,
    test_connect_to_storage,
):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluator._evaluation_process_dict[1] = get_evaluation_process_info()
        evaluator._evaluation_dict[1] = None

        response = evaluator.get_evaluation_status(EvaluationStatusRequest(evaluation_id=1), None)
        assert response.valid
        assert response.is_running
        assert response.blocked
        assert not response.state_available
        assert not response.HasField("exception")
        assert not (response.HasField("batches_seen") or response.HasField("samples_seen"))
        test_check_for_evaluation_exception.assert_not_called()


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
@patch.object(mp.Process, "is_alive", return_value=False)
@patch.object(EvaluatorGRPCServicer, "_check_for_evaluation_exception", return_value="exception")
@patch.object(EvaluatorGRPCServicer, "_get_status")
def test_get_evaluation_status_finished_with_exception(
    test_get_status,
    test_check_for_evaluation_exception,
    test_is_alive,
    test_connect_to_model_storage,
    test_connect_to_storage,
):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        evaluator._evaluation_process_dict[2] = get_evaluation_process_info()
        evaluator._evaluation_dict[2] = None

        response = evaluator.get_evaluation_status(EvaluationStatusRequest(evaluation_id=2), None)
        assert response.valid
        assert not response.is_running
        assert not response.blocked
        assert not response.state_available
        assert not (response.HasField("batches_seen") or response.HasField("samples_seen"))
        assert response.HasField("exception")
        assert response.exception == "exception"
        test_get_status.assert_not_called()


@patch.object(EvaluatorGRPCServicer, "connect_to_storage", return_value=DummyStorageStub())
@patch.object(EvaluatorGRPCServicer, "connect_to_model_storage", return_value=DummyModelStorageStub())
def test_get_evaluation_status(test_connect_to_model_storage, test_connect_to_storage):
    with tempfile.TemporaryDirectory() as modyn_temp:
        evaluator = EvaluatorGRPCServicer(get_modyn_config(), pathlib.Path(modyn_temp))
        state_dict = {"num_batches": 10, "num_samples": 100}

        evaluation_process_info = get_evaluation_process_info()
        evaluator._evaluation_process_dict[1] = evaluation_process_info
        evaluation_process_info.status_response_queue.put(state_dict)
        num_batches, num_samples = evaluator._get_status(1)
        assert num_batches == state_dict["num_batches"]
        assert num_samples == state_dict["num_samples"]

        timeout = 5
        elapsed = 0

        while True:
            if not platform.system() == "Darwin":
                if evaluation_process_info.status_query_queue.qsize() == 1:
                    break
            else:
                if not evaluation_process_info.status_query_queue.empty():
                    break

            sleep(0.1)
            elapsed += 0.1

            if elapsed >= timeout:
                raise AssertionError("Did not reach desired queue state after 5 seconds.")

        if not platform.system() == "Darwin":
            assert evaluation_process_info.status_response_queue.qsize() == 0
        else:
            assert evaluation_process_info.status_response_queue.empty()

        query = evaluation_process_info.status_query_queue.get()
        assert query == EvaluatorMessages.STATUS_QUERY_MESSAGE


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
        assert response.valid
        assert len(response.evaluation_data) == 0


@patch.object(Accuracy, "get_evaluation_result", return_value=0.5)
@patch.object(F1Score, "get_evaluation_result", return_value=0.75)
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
    with tempfile.TemporaryDirectory() as temp:
        config = get_modyn_config()
        evaluator = EvaluatorGRPCServicer(config, pathlib.Path(temp))
        evaluator._evaluation_dict[1] = get_evaluation_info(1, pathlib.Path(temp) / "trained_model.modyn", config)

        assert len(evaluator._evaluation_dict[1].metrics) == 2
        assert isinstance(evaluator._evaluation_dict[1].metrics[0], Accuracy)
        assert isinstance(evaluator._evaluation_dict[1].metrics[1], F1Score)

        evaluation_process_info = get_evaluation_process_info()
        evaluator._evaluation_process_dict[1] = evaluation_process_info
        for metric in evaluator._evaluation_dict[1].metrics:
            evaluation_process_info.metric_result_queue.put((metric.get_name(), metric.get_evaluation_result()))

        timeout = 5
        elapsed = 0

        while True:
            if not platform.system() == "Darwin":
                if evaluation_process_info.metric_result_queue.qsize() == 2:
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
        assert len(response.evaluation_data) == 2
        assert response.evaluation_data[0].metric == Accuracy.get_name()
        assert response.evaluation_data[0].result == 0.5
        test_acc.assert_called_once()
        assert response.evaluation_data[1].metric == F1Score.get_name()
        assert response.evaluation_data[1].result == 0.75
        test_f1.assert_called_once()
