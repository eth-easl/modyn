# pylint: disable=unused-argument,no-value-for-parameter,no-name-in-module
import json
from unittest.mock import patch

import grpc
import pytest

from modyn.config.schema.pipeline import EvalDataConfig
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    EvaluationIntervalData,
    EvaluationResultRequest,
    EvaluationResultResponse,
    EvaluationStatusResponse,
    SingleMetricResult,
)
from modyn.evaluator.internal.grpc.generated.evaluator_pb2_grpc import EvaluatorStub
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    DataInformRequest,
    DataInformResponse,
    GetNumberOfSamplesRequest,
    JsonString,
    NumberOfSamplesResponse,
    TriggerResponse,
)
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableResponse,
    GetDataInIntervalRequest,
    GetDataInIntervalResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2_grpc import TrainerServerStub


def noop_constructor_mock(self, channel: grpc.Channel) -> None:
    pass


def get_simple_config() -> dict:
    return {
        "storage": {"hostname": "test", "port": 42},
        "trainer_server": {"hostname": "localhost", "port": 42, "ftp_port": 1337},
        "selector": {"hostname": "test", "port": 42},
        "evaluator": {"hostname": "test", "port": 42},
    }


def get_minimal_dataset_config() -> EvalDataConfig:
    return EvalDataConfig.model_validate(
        {
            "dataset_id": "MNIST_eval",
            "bytes_parser_function": "def bytes_parser_function(data: bytes) -> bytes:\n\treturn data",
            "dataloader_workers": 2,
            "batch_size": 64,
            "metrics": [{"name": "Accuracy"}],
        }
    )


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(EvaluatorStub, "__init__", noop_constructor_mock)
@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def get_non_connecting_handler(*args) -> GRPCHandler:
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()
    return handler


def test_init():
    GRPCHandler(get_simple_config())


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(EvaluatorStub, "__init__", noop_constructor_mock)
@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_cluster_connection(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()

    assert handler.connected_to_storage
    assert handler.connected_to_selector
    assert handler.connected_to_evaluator
    assert handler.storage is not None
    assert handler.selector is not None
    assert handler.evaluator is not None


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(EvaluatorStub, "__init__", noop_constructor_mock)
@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_storage(test_insecure_channel, test_connection_established):
    handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_storage

    handler.init_storage()

    assert handler.connected_to_storage
    assert handler.storage is not None


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(EvaluatorStub, "__init__", noop_constructor_mock)
@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=False)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_storage_throws(test_insecure_channel, test_connection_established):
    handler = None
    handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_storage

    with pytest.raises(ConnectionError):
        handler.init_storage()


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(EvaluatorStub, "__init__", noop_constructor_mock)
@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_selector(test_insecure_channel, test_connection_established):
    handler = None
    handler = GRPCHandler(get_simple_config())  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_selector

    handler.init_selector()

    assert handler.connected_to_selector
    assert handler.selector is not None


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_dataset_available(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()

    assert handler.storage is not None

    with patch.object(
        handler.storage, "CheckAvailability", return_value=DatasetAvailableResponse(available=True)
    ) as avail_method:
        assert handler.dataset_available("id")
        avail_method.assert_called_once()

    with patch.object(
        handler.storage, "CheckAvailability", return_value=DatasetAvailableResponse(available=False)
    ) as avail_method:
        assert not handler.dataset_available("id")
        avail_method.assert_called_once()


def test_get_new_data_since_throws():
    handler = get_non_connecting_handler()
    handler.connected_to_storage = False
    with pytest.raises(ConnectionError):
        list(handler.get_new_data_since("dataset_id", 0))


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_get_new_data_since(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()

    with patch.object(handler.storage, "GetNewDataSince") as mock:
        mock.return_value = [GetNewDataSinceResponse(keys=[0, 1], timestamps=[41, 42], labels=[0, 1])]

        result = [data for data, _ in handler.get_new_data_since("test_dataset", 21)]

        assert result == [[(0, 41, 0), (1, 42, 1)]]
        mock.assert_called_once_with(GetNewDataSinceRequest(dataset_id="test_dataset", timestamp=21))


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_get_new_data_since_batched(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()

    with patch.object(handler.storage, "GetNewDataSince") as mock:
        mock.return_value = [
            GetNewDataSinceResponse(keys=[0, 1], timestamps=[41, 42], labels=[0, 1]),
            GetNewDataSinceResponse(keys=[2, 3], timestamps=[42, 43], labels=[0, 1]),
        ]

        result = [data for data, _ in handler.get_new_data_since("test_dataset", 21)]

        assert result == [[(0, 41, 0), (1, 42, 1)], [(2, 42, 0), (3, 43, 1)]]
        mock.assert_called_once_with(GetNewDataSinceRequest(dataset_id="test_dataset", timestamp=21))


def test_get_data_in_interval_throws():
    handler = get_non_connecting_handler()
    handler.connected_to_storage = False
    with pytest.raises(ConnectionError):
        list(handler.get_data_in_interval("dataset_id", 0, 1))


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_get_data_in_interval(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()

    with patch.object(handler.storage, "GetDataInInterval") as mock:
        mock.return_value = [GetDataInIntervalResponse(keys=[0, 1], timestamps=[41, 42], labels=[0, 1])]

        result = [data for data, _ in handler.get_data_in_interval("test_dataset", 21, 45)]

        assert result == [[(0, 41, 0), (1, 42, 1)]]
        mock.assert_called_once_with(
            GetDataInIntervalRequest(dataset_id="test_dataset", start_timestamp=21, end_timestamp=45)
        )


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_get_data_in_interval_batched(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()

    with patch.object(handler.storage, "GetDataInInterval") as mock:
        mock.return_value = [
            GetDataInIntervalResponse(keys=[0, 1], timestamps=[41, 42], labels=[0, 1]),
            GetDataInIntervalResponse(keys=[2, 3], timestamps=[42, 43], labels=[0, 1]),
        ]

        result = [data for data, _ in handler.get_data_in_interval("test_dataset", 21, 45)]

        assert result == [[(0, 41, 0), (1, 42, 1)], [(2, 42, 0), (3, 43, 1)]]
        mock.assert_called_once_with(
            GetDataInIntervalRequest(dataset_id="test_dataset", start_timestamp=21, end_timestamp=45)
        )


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_inform_selector(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()

    with patch.object(handler.selector, "inform_data") as mock:
        mock.return_value = DataInformResponse(log=JsonString(value='{"1": 2}'))

        log = handler.inform_selector(42, [(10, 42, 0), (11, 43, 1)])

        mock.assert_called_once_with(
            DataInformRequest(pipeline_id=42, keys=[10, 11], timestamps=[42, 43], labels=[0, 1])
        )

        assert log == {"1": 2}


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_inform_selector_and_trigger(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()

    with patch.object(handler.selector, "inform_data_and_trigger") as mock:
        mock.return_value = TriggerResponse(trigger_id=12, log=JsonString(value='{"1": 2}'))

        trigger_id, log = handler.inform_selector_and_trigger(42, [(10, 42, 0), (11, 43, 1)])
        assert trigger_id == 12
        assert log == {"1": 2}

        mock.assert_called_once_with(
            DataInformRequest(pipeline_id=42, keys=[10, 11], timestamps=[42, 43], labels=[0, 1])
        )

    # Test empty trigger
    with patch.object(handler.selector, "inform_data_and_trigger") as mock:
        mock.return_value = TriggerResponse(trigger_id=13, log=JsonString(value='{"1": 2}'))
        trigger_id, log = handler.inform_selector_and_trigger(42, [])
        assert 13 == trigger_id
        assert log == {"1": 2}

        mock.assert_called_once_with(DataInformRequest(pipeline_id=42, keys=[], timestamps=[], labels=[]))


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_get_number_of_samples(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()
    assert handler.selector is not None

    with patch.object(
        handler.selector, "get_number_of_samples", return_value=NumberOfSamplesResponse(num_samples=42)
    ) as samples_method:
        assert handler.get_number_of_samples(12, 13)
        samples_method.assert_called_once_with(GetNumberOfSamplesRequest(pipeline_id=12, trigger_id=13))


def test_stop_training_at_trainer_server():
    handler = get_non_connecting_handler()
    training_id = 42

    # TODO(#78,#130): implement a real test when func is implemented.
    handler.stop_training_at_trainer_server(training_id)


@pytest.mark.parametrize("tokenizer", [None, "DistilBertTokenizerTransform"])
def test_prepare_evaluation_request(tokenizer: str):
    dataset_config = get_minimal_dataset_config()
    dataset_config.tokenizer = tokenizer
    intervals = [
        (None, None),
        (None, 42),
        (42, None),
        (42, 43),
    ]
    request = GRPCHandler.prepare_evaluation_request(dataset_config.model_dump(by_alias=True), 23, "cpu", intervals)

    assert request.model_id == 23
    assert request.device == "cpu"
    assert request.batch_size == 64
    assert request.dataset_info.dataset_id == "MNIST_eval"
    assert request.dataset_info.num_dataloaders == 2
    assert json.loads(str(request.metrics[0].value))["name"] == "Accuracy"
    if tokenizer:
        assert request.HasField("tokenizer")
        assert request.tokenizer.value == "DistilBertTokenizerTransform"
    else:
        assert not request.HasField("tokenizer")
    assert len(request.dataset_info.evaluation_intervals) == len(intervals)
    for expected_interval, interval in zip(intervals, request.dataset_info.evaluation_intervals):
        expected_start_ts = expected_interval[0]
        expected_end_ts = expected_interval[1]
        if expected_start_ts:
            assert interval.start_timestamp == expected_start_ts
        else:
            assert not interval.HasField("start_timestamp")

        if expected_end_ts:
            assert interval.end_timestamp == expected_end_ts
        else:
            assert not interval.HasField("end_timestamp")


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_wait_for_evaluation_completion(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()
    assert handler.evaluator is not None

    with patch.object(handler.evaluator, "get_evaluation_status") as status_method:
        with patch.object(handler.evaluator, "cleanup_evaluations") as _:
            status_method.side_effect = [
                EvaluationStatusResponse(valid=True, is_running=True),
                EvaluationStatusResponse(valid=True, is_running=False),
            ]

            assert handler.wait_for_evaluation_completion(10)
            assert status_method.call_count == 2

            status_method.reset_mock()
            status_method.side_effect = [
                EvaluationStatusResponse(valid=True, exception="Some error"),
                EvaluationStatusResponse(valid=True, is_running=False),
            ]
            assert not handler.wait_for_evaluation_completion(10)
            assert status_method.call_count == 1


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch("modyn.common.grpc.grpc_helpers.grpc_connection_established", return_value=True)
def test_get_evaluation_results(*args):
    handler = GRPCHandler(get_simple_config())
    handler.init_cluster_connection()
    assert handler.evaluator is not None

    evaluation_data = [
        EvaluationIntervalData(
            evaluation_data=[
                SingleMetricResult(metric="Accuracy", result=0.42),
                SingleMetricResult(metric="Loss", result=0.13),
            ]
        ),
        EvaluationIntervalData(
            evaluation_data=[
                SingleMetricResult(metric="Accuracy", result=0.43),
                SingleMetricResult(metric="Loss", result=0.14),
            ]
        ),
    ]
    with patch.object(handler.evaluator, "get_evaluation_result") as result_method:
        result_method.return_value = EvaluationResultResponse(
            evaluation_results=evaluation_data,
            valid=True,
        )
        res = handler.get_evaluation_results(10)
        result_method.assert_called_once_with(EvaluationResultRequest(evaluation_id=10))
        assert res == evaluation_data
        result_method.reset_mock()
        result_method.return_value = EvaluationResultResponse(
            valid=False,
        )
        with pytest.raises(RuntimeError):
            handler.get_evaluation_results(15)
