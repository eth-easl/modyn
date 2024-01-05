# pylint: disable=unused-argument,no-value-for-parameter,no-name-in-module
import json
import multiprocessing as mp
import pathlib
import tempfile
from unittest.mock import patch

import grpc
import pytest
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    EvaluateModelResponse,
    EvaluationData,
    EvaluationResultRequest,
    EvaluationResultResponse,
    EvaluationStatusResponse,
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
from modyn.supervisor.internal.evaluation_result_writer import JsonResultWriter
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.utils import EvaluationStatusReporter
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import JsonString as TrainerJsonString
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    StartTrainingResponse,
    StoreFinalModelRequest,
    StoreFinalModelResponse,
    TrainerAvailableResponse,
    TrainingStatusResponse,
)
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


def get_minimal_pipeline_config() -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "training": {
            "gpus": 1,
            "device": "cpu",
            "amp": False,
            "dataloader_workers": 1,
            "initial_model": "random",
            "batch_size": 42,
            "optimizers": [
                {"name": "default", "algorithm": "SGD", "source": "PyTorch", "param_groups": [{"module": "model"}]},
            ],
            "optimization_criterion": {"name": "CrossEntropyLoss"},
            "checkpointing": {"activated": False},
            "selection_strategy": {"name": "NewDataStrategy"},
        },
        "data": {"dataset_id": "test", "bytes_parser_function": "def bytes_parser_function(x):\n\treturn x"},
        "trigger": {"id": "DataAmountTrigger", "trigger_config": {"data_points_for_trigger": 1}},
        "evaluation": {
            "device": "cpu",
            "datasets": [
                {
                    "dataset_id": "MNIST_eval",
                    "bytes_parser_function": "def bytes_parser_function(data: bytes) -> bytes:\n\treturn data",
                    "dataloader_workers": 2,
                    "batch_size": 64,
                    "metrics": [{"name": "Accuracy"}],
                }
            ],
        },
    }


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(EvaluatorStub, "__init__", noop_constructor_mock)
@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def get_non_connecting_handler(insecure_channel, init) -> GRPCHandler:
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()
    return handler


def test_init():
    GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(EvaluatorStub, "__init__", noop_constructor_mock)
@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_cluster_connection(test_insecure_channel, test_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()

    assert handler.connected_to_storage
    assert handler.connected_to_trainer_server
    assert handler.storage is not None


@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch.object(TrainerServerStub, "__init__", noop_constructor_mock)
@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(EvaluatorStub, "__init__", noop_constructor_mock)
@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_storage(test_insecure_channel, test_connection_established):
    handler = None
    handler = GRPCHandler(
        get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue()
    )  # don't call init storage in constructor

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
    handler = GRPCHandler(
        get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue()
    )  # don't call init storage in constructor

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
    handler = GRPCHandler(
        get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue()
    )  # don't call init storage in constructor

    assert handler is not None
    assert not handler.connected_to_selector

    handler.init_selector()

    assert handler.connected_to_selector
    assert handler.selector is not None


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_dataset_available(test_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
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
def test_get_new_data_since(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()

    with patch.object(handler.storage, "GetNewDataSince") as mock:
        mock.return_value = [GetNewDataSinceResponse(keys=[0, 1], timestamps=[41, 42], labels=[0, 1])]

        result = [data for data, _ in handler.get_new_data_since("test_dataset", 21)]

        assert result == [[(0, 41, 0), (1, 42, 1)]]
        mock.assert_called_once_with(GetNewDataSinceRequest(dataset_id="test_dataset", timestamp=21))


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_get_new_data_since_batched(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
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
def test_get_data_in_interval(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()

    with patch.object(handler.storage, "GetDataInInterval") as mock:
        mock.return_value = [GetDataInIntervalResponse(keys=[0, 1], timestamps=[41, 42], labels=[0, 1])]

        result = [data for data, _ in handler.get_data_in_interval("test_dataset", 21, 45)]

        assert result == [[(0, 41, 0), (1, 42, 1)]]
        mock.assert_called_once_with(
            GetDataInIntervalRequest(dataset_id="test_dataset", start_timestamp=21, end_timestamp=45)
        )


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_get_data_in_interval_batched(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
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
def test_inform_selector(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()

    with patch.object(handler.selector, "inform_data") as mock:
        mock.return_value = DataInformResponse(log=JsonString(value='{"1": 2}'))

        log = handler.inform_selector(42, [(10, 42, 0), (11, 43, 1)])

        mock.assert_called_once_with(
            DataInformRequest(pipeline_id=42, keys=[10, 11], timestamps=[42, 43], labels=[0, 1])
        )

        assert log == {"1": 2}


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_inform_selector_and_trigger(test_grpc_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
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
def test_trainer_server_available(test_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()
    assert handler.trainer_server is not None

    with patch.object(
        handler.trainer_server, "trainer_available", return_value=TrainerAvailableResponse(available=True)
    ) as avail_method:
        assert handler.trainer_server_available()
        avail_method.assert_called_once()


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_get_number_of_samples(test_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
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


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_start_training(test_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()
    assert handler.trainer_server is not None

    pipeline_id = 42
    trigger_id = 21
    pipeline_config = get_minimal_pipeline_config()

    with patch.object(
        handler.trainer_server,
        "start_training",
        return_value=StartTrainingResponse(training_started=True, training_id=42),
    ) as avail_method:
        assert handler.start_training(pipeline_id, trigger_id, pipeline_config, None) == 42
        avail_method.assert_called_once()


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_wait_for_training_completion(test_connection_established):
    # This test primarily checks whether we terminate.
    with patch.object(GRPCHandler, "get_number_of_samples", return_value=22):
        with patch.object(GRPCHandler, "get_status_bar_scale", return_value=100):
            handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
            handler.init_cluster_connection()
            assert handler.trainer_server is not None

            with patch.object(
                handler.trainer_server,
                "get_training_status",
                return_value=TrainingStatusResponse(
                    valid=True,
                    blocked=False,
                    exception=None,
                    state_available=False,
                    is_running=False,
                    log=TrainerJsonString(value='{"a": 1}'),
                ),
            ) as avail_method:
                log = handler.wait_for_training_completion(42, 21, 22)
                avail_method.assert_called_once()
                assert log == {"a": 1}


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_store_trained_model(test_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()
    assert handler.trainer_server is not None

    res = StoreFinalModelResponse(valid_state=True, model_id=42)

    with patch.object(handler.trainer_server, "store_final_model", return_value=res) as get_method:
        model_id = handler.store_trained_model(21)
        get_method.assert_called_once_with(StoreFinalModelRequest(training_id=21))
        assert model_id == 42


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_start_evaluation(test_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()
    assert handler.evaluator is not None

    model_id = 10
    pipeline_config = get_minimal_pipeline_config()

    with patch.object(
        handler.evaluator,
        "evaluate_model",
        return_value=EvaluateModelResponse(evaluation_started=True, evaluation_id=12, dataset_size=1000),
    ) as avail_method:
        evaluations = handler.start_evaluation(model_id, pipeline_config)

        assert len(evaluations) == 1
        assert evaluations[12].dataset_id == "MNIST_eval"
        assert evaluations[12].dataset_size == 1000
        avail_method.assert_called_once()


def test__prepare_evaluation_request():
    pipeline_config = get_minimal_pipeline_config()
    request = GRPCHandler._prepare_evaluation_request(pipeline_config["evaluation"]["datasets"][0], 23, "cpu")

    assert request.model_id == 23
    assert request.device == "cpu"
    assert request.batch_size == 64
    assert request.dataset_info.dataset_id == "MNIST_eval"
    assert request.dataset_info.num_dataloaders == 2
    assert request.metrics[0].name == "Accuracy"
    assert request.metrics[0].config.value == "{}"


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_wait_for_evaluation_completion(test_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()
    assert handler.evaluator is not None

    evaluations = {
        1: EvaluationStatusReporter(
            dataset_id="MNIST_small", dataset_size=1000, evaluation_id=1, eval_status_queue=handler.eval_status_queue
        ),
        2: EvaluationStatusReporter(
            dataset_id="MNIST_big", dataset_size=5000, evaluation_id=2, eval_status_queue=handler.eval_status_queue
        ),
        3: EvaluationStatusReporter(
            dataset_id="MNIST_large", dataset_size=10000, evaluation_id=3, eval_status_queue=handler.eval_status_queue
        ),
    }

    with patch.object(handler.evaluator, "get_evaluation_status") as status_method:
        status_method.side_effect = [
            EvaluationStatusResponse(valid=False),
            EvaluationStatusResponse(valid=True, blocked=True),
            EvaluationStatusResponse(
                valid=True, blocked=False, is_running=True, state_available=True, batches_seen=10, samples_seen=5000
            ),
            EvaluationStatusResponse(valid=True, blocked=False, exception="Error"),
            EvaluationStatusResponse(valid=True, blocked=False, is_running=False, state_available=False),
        ]
        handler.wait_for_evaluation_completion(10, evaluations)
        assert status_method.call_count == 5

        # from call get args (call[0]) then get first argument
        called_ids = [call[0][0].evaluation_id for call in status_method.call_args_list]
        assert called_ids == [1, 2, 3, 2, 3]


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_store_evaluation_results(test_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()
    assert handler.evaluator is not None

    res = EvaluationResultResponse(
        valid=True,
        evaluation_data=[EvaluationData(metric="Accuracy", result=0.5), EvaluationData(metric="F1-score", result=0.75)],
    )

    evaluations = {
        10: EvaluationStatusReporter(
            dataset_id="MNIST_small", dataset_size=1000, evaluation_id=10, eval_status_queue=handler.eval_status_queue
        ),
        15: EvaluationStatusReporter(
            dataset_id="MNIST_large", dataset_size=5000, evaluation_id=15, eval_status_queue=handler.eval_status_queue
        ),
    }

    with tempfile.TemporaryDirectory() as path:
        with patch.object(handler.evaluator, "get_evaluation_result", return_value=res) as get_method:
            eval_dir = pathlib.Path(path)
            handler.store_evaluation_results([JsonResultWriter(5, 3, eval_dir)], evaluations)
            assert get_method.call_count == 2

            called_ids = [call[0][0].evaluation_id for call in get_method.call_args_list]
            assert called_ids == [10, 15]

            file_path = eval_dir / f"{5}_{3}.eval"
            assert file_path.exists() and file_path.is_file()

            with open(file_path, "r", encoding="utf-8") as eval_file:
                evaluation_results = json.load(eval_file)
                assert evaluation_results == json.loads(
                    """{
                    "datasets": [
                        {
                            "MNIST_small": {
                                "dataset_size": 1000,
                                "metrics": [
                                    {
                                        "name": "Accuracy",
                                        "result": 0.5
                                    },
                                    {
                                        "name": "F1-score",
                                        "result": 0.75
                                    }
                                ]
                            }
                        },
                        {
                            "MNIST_large": {
                                "dataset_size": 5000,
                                "metrics": [
                                    {
                                        "name": "Accuracy",
                                        "result": 0.5
                                    },
                                    {
                                        "name": "F1-score",
                                        "result": 0.75
                                    }
                                ]
                            }
                        }
                    ]
                }"""
                )


@patch("modyn.supervisor.internal.grpc_handler.grpc_connection_established", return_value=True)
def test_store_evaluation_results_invalid(test_connection_established):
    handler = GRPCHandler(get_simple_config(), mp.Queue(), mp.Queue(), mp.Queue())
    handler.init_cluster_connection()
    assert handler.evaluator is not None

    res = EvaluationResultResponse(valid=False)

    evaluations = {
        10: EvaluationStatusReporter(
            dataset_id="MNIST_small", dataset_size=1000, evaluation_id=10, eval_status_queue=handler.eval_status_queue
        )
    }

    with tempfile.TemporaryDirectory() as path:
        with patch.object(handler.evaluator, "get_evaluation_result", return_value=res) as get_method:
            eval_dir = pathlib.Path(path)

            handler.store_evaluation_results([JsonResultWriter(5, 3, eval_dir)], evaluations)
            get_method.assert_called_with(EvaluationResultRequest(evaluation_id=10))

            file_path = eval_dir / f"{5}_{3}.eval"
            assert file_path.exists() and file_path.is_file()

            with open(file_path, "r", encoding="utf-8") as eval_file:
                evaluation_results = json.load(eval_file)
                assert evaluation_results["datasets"] == []
