# pylint: disable=unused-argument,redefined-outer-name
import multiprocessing as mp
import os
import pathlib
import shutil
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock, call, patch

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluateModelResponse, EvaluationNotStartReason
from modyn.supervisor.internal.eval_strategies.periodical_eval_strategy import PeriodicalEvalStrategy
from modyn.supervisor.internal.evaluation_result_writer import (
    AbstractEvaluationResultWriter,
    JsonResultWriter,
    TensorboardResultWriter,
)
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.pipeline_executor import PipelineExecutor, execute_pipeline
from modyn.supervisor.internal.utils.evaluation_status_reporter import EvaluationStatusReporter

EVALUATION_DIRECTORY: pathlib.Path = pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir"
SUPPORTED_EVAL_RESULT_WRITERS: dict = {"json": JsonResultWriter, "tensorboard": TensorboardResultWriter}
START_TIMESTAMP = 21
PIPELINE_ID = 42
EVAL_ID = 42
EXCEPTION_QUEUE = mp.Queue()
PIPELINE_STATUS_QUEUE = mp.Queue()
TRAINING_STATUS_QUEUE = mp.Queue()
EVAL_STATUS_QUEUE = mp.Queue()


def get_minimal_training_config() -> dict:
    return {
        "gpus": 1,
        "device": "cpu",
        "dataloader_workers": 1,
        "use_previous_model": True,
        "initial_model": "random",
        "learning_rate": 0.1,
        "batch_size": 42,
        "optimizers": [
            {"name": "default1", "algorithm": "SGD", "source": "PyTorch", "param_groups": [{"module": "model"}]},
        ],
        "optimization_criterion": {"name": "CrossEntropyLoss"},
        "checkpointing": {"activated": False},
        "selection_strategy": {"name": "NewDataStrategy", "maximum_keys_in_memory": 10},
    }


def get_minimal_evaluation_config() -> dict:
    return {
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
    }


def get_minimal_pipeline_config() -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "model_storage": {"full_model_strategy": {"name": "PyTorchFullModel"}},
        "training": get_minimal_training_config(),
        "data": {"dataset_id": "test", "bytes_parser_function": "def bytes_parser_function(x):\n\treturn x"},
        "trigger": {"id": "DataAmountTrigger", "trigger_config": {"data_points_for_trigger": 1}},
    }


def get_minimal_eval_strategies_config() -> dict:
    return {
        "name": "PeriodicalEvalStrategy",
        "config": {
            "eval_every": "100s",
            "eval_start_from": 0,
            "eval_end_at": 300,
        },
    }


def get_minimal_system_config() -> dict:
    return {}


def noop_constructor_mock(
    self,
    start_timestamp: int,
    pipeline_id: int,
    modyn_config: dict,
    pipeline_config: dict,
    eval_directory: str,
    supervisor_supported_eval_result_writers: dict,
    pipeline_status_queue: mp.Queue,
    training_status_queue: mp.Queue,
    eval_status_queue: mp.Queue,
    start_replay_at: Optional[int] = None,
    stop_replay_at: Optional[int] = None,
    maximum_triggers: Optional[int] = None,
    evaluation_matrix: bool = False,
) -> None:
    pass


def sleep_mock(duration: int):
    raise KeyboardInterrupt


def setup():
    if EVALUATION_DIRECTORY.is_dir():
        shutil.rmtree(EVALUATION_DIRECTORY)
    EVALUATION_DIRECTORY.mkdir(0o777)


def teardown():
    shutil.rmtree(EVALUATION_DIRECTORY)


def get_non_connecting_pipeline_executor() -> PipelineExecutor:
    pipeline_executor = PipelineExecutor(
        START_TIMESTAMP,
        PIPELINE_ID,
        get_minimal_system_config(),
        get_minimal_pipeline_config(),
        str(EVALUATION_DIRECTORY),
        SUPPORTED_EVAL_RESULT_WRITERS,
        PIPELINE_STATUS_QUEUE,
        TRAINING_STATUS_QUEUE,
        EVAL_STATUS_QUEUE,
    )
    return pipeline_executor


def test_initialization() -> None:
    get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter


def test_get_dataset_selector_batch_size_given():
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter

    pe.pipeline_config = get_minimal_pipeline_config()
    pe.modyn_config = {
        "storage": {
            "datasets": [{"name": "test", "selector_batch_size": 2048}, {"name": "test1", "selector_batch_size": 128}]
        }
    }
    pe.get_dataset_selector_batch_size()
    assert pe._selector_batch_size == 2048


def test_get_dataset_selector_batch_size_not_given():
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter

    pe.pipeline_config = get_minimal_pipeline_config()
    pe.modyn_config = {"storage": {"datasets": [{"name": "test"}]}}
    pe.get_dataset_selector_batch_size()
    assert pe._selector_batch_size == 128


def test_shutdown_trainer():
    # TODO(MaxiBoether): implement
    pass


@patch.object(GRPCHandler, "get_new_data_since", return_value=[([(10, 42, 0), (11, 43, 1)], {})])
@patch.object(PipelineExecutor, "_handle_new_data", return_value=False, side_effect=KeyboardInterrupt)
def test_wait_for_new_data(test__handle_new_data: MagicMock, test_get_new_data_since: MagicMock):
    # This is a simple test and does not the inclusivity filtering!
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter

    pe.wait_for_new_data(21)
    test_get_new_data_since.assert_called_once_with("test", 21)
    test__handle_new_data.assert_called_once_with([(10, 42, 0), (11, 43, 1)])


@patch.object(GRPCHandler, "get_new_data_since", return_value=[([(10, 42, 0)], {}), ([(11, 43, 1)], {})])
@patch.object(PipelineExecutor, "_handle_new_data", return_value=False, side_effect=[None, KeyboardInterrupt])
def test_wait_for_new_data_batched(test__handle_new_data: MagicMock, test_get_new_data_since: MagicMock):
    # This is a simple test and does not the inclusivity filtering!
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter

    pe.wait_for_new_data(21)
    test_get_new_data_since.assert_called_once_with("test", 21)

    expected_calls = [
        call([(10, 42, 0)]),
        call([(11, 43, 1)]),
    ]

    assert test__handle_new_data.call_args_list == expected_calls


def test_wait_for_new_data_filtering():
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter

    mocked__handle_new_data_return_vals = [True, True, KeyboardInterrupt]
    mocked_get_new_data_since = [
        [([(10, 42, 0), (11, 43, 0), (12, 43, 1)], {})],
        [([(11, 43, 0), (12, 43, 1), (13, 43, 2), (14, 45, 3)], {})],
        [([], {})],
        ValueError,
    ]

    handle_mock: MagicMock
    with patch.object(pe, "_handle_new_data", side_effect=mocked__handle_new_data_return_vals) as handle_mock:
        get_new_data_mock: MagicMock
        with patch.object(pe.grpc, "get_new_data_since", side_effect=mocked_get_new_data_since) as get_new_data_mock:
            pe.wait_for_new_data(21)

            assert handle_mock.call_count == 3
            assert get_new_data_mock.call_count == 3

            expected_handle_mock_arg_list = [
                call([(10, 42, 0), (11, 43, 0), (12, 43, 1)]),
                call([(13, 43, 2), (14, 45, 3)]),
                call([]),
            ]
            assert handle_mock.call_args_list == expected_handle_mock_arg_list

            expected_get_new_data_arg_list = [call("test", 21), call("test", 43), call("test", 45)]
            assert get_new_data_mock.call_args_list == expected_get_new_data_arg_list


def test_wait_for_new_data_filtering_batched():
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter

    mocked__handle_new_data_return_vals = [True, True, True, True, True, KeyboardInterrupt]
    mocked_get_new_data_since = [
        [([(10, 42, 0), (11, 43, 0)], {}), ([(12, 43, 1)], {})],
        [([(11, 43, 0)], {}), ([(12, 43, 1), (13, 43, 2)], {}), ([(14, 45, 3)], {})],
        [([], {})],
        ValueError,
    ]

    handle_mock: MagicMock
    with patch.object(pe, "_handle_new_data", side_effect=mocked__handle_new_data_return_vals) as handle_mock:
        get_new_data_mock: MagicMock
        with patch.object(pe.grpc, "get_new_data_since", side_effect=mocked_get_new_data_since) as get_new_data_mock:
            pe.wait_for_new_data(21)

            assert handle_mock.call_count == 6
            assert get_new_data_mock.call_count == 3

            expected_handle_mock_arg_list = [
                call([(10, 42, 0), (11, 43, 0)]),
                call([(12, 43, 1)]),
                call([]),
                call([(13, 43, 2)]),
                call([(14, 45, 3)]),
                call([]),
            ]
            assert handle_mock.call_args_list == expected_handle_mock_arg_list

            expected_get_new_data_arg_list = [call("test", 21), call("test", 43), call("test", 45)]
            assert get_new_data_mock.call_args_list == expected_get_new_data_arg_list


def test__handle_new_data_with_batch():
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe._selector_batch_size = 3
    new_data = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (15, 6), (16, 7), (17, 8)]

    batch_mock: MagicMock
    with patch.object(pe, "_handle_new_data_batch") as batch_mock:
        pe._handle_new_data(new_data)
        expected_handle_new_data_batch_arg_list = [
            call([(10, 1), (11, 2), (12, 3)]),
            call([(13, 4), (14, 5), (15, 6)]),
            call([(16, 7), (17, 8)]),
        ]
        assert batch_mock.call_args_list == expected_handle_new_data_batch_arg_list


def test__handle_new_data_with_large_batch():
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    new_data = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (15, 6), (16, 7), (17, 8)]

    batch_mock: MagicMock
    with patch.object(pe, "_handle_new_data_batch") as batch_mock:
        pe._handle_new_data(new_data)
        expected_handle_new_data_batch_arg_list = [call(new_data)]
        assert batch_mock.call_args_list == expected_handle_new_data_batch_arg_list


def test__handle_new_data():
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter

    pe._selector_batch_size = 2
    batching_return_vals = [False, True, False]
    new_data = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5)]

    batch_mock: MagicMock
    with patch.object(pe, "_handle_new_data_batch", side_effect=batching_return_vals) as batch_mock:
        result = pe._handle_new_data(new_data)
        assert result

        expected_handle_new_data_batch_arg_list = [
            call([(10, 1), (11, 2)]),
            call([(12, 3), (13, 4)]),
            call([(14, 5)]),
        ]
        assert batch_mock.call_count == 3
        assert batch_mock.call_args_list == expected_handle_new_data_batch_arg_list


@patch.object(GRPCHandler, "inform_selector")
def test__handle_new_data_batch_no_triggers(test_inform_selector: MagicMock):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.pipeline_id = 42
    batch = [(10, 1), (11, 2)]

    with patch.object(pe.trigger, "inform", return_value=[]) as inform_mock:
        assert not pe._handle_new_data_batch(batch)

        inform_mock.assert_called_once_with(batch)
        test_inform_selector.assert_called_once_with(42, batch)


@patch.object(PipelineExecutor, "_run_training")
@patch.object(GRPCHandler, "inform_selector_and_trigger")
@patch.object(GRPCHandler, "inform_selector")
@patch.object(GRPCHandler, "get_number_of_samples")
def test__handle_triggers_within_batch(
    test_get_number_of_samples: MagicMock,
    test_inform_selector: MagicMock,
    test_inform_selector_and_trigger: MagicMock,
    test__run_training: MagicMock,
):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.pipeline_id = 42
    batch = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (15, 6), (16, 7)]
    triggering_indices = [1, 3, 5]
    trigger_ids = [(0, {}), (1, {}), (2, {})]
    test_inform_selector_and_trigger.side_effect = trigger_ids
    test_inform_selector.return_value = {}
    test_get_number_of_samples.return_value = len(batch)

    pe._handle_triggers_within_batch(batch, triggering_indices)

    inform_selector_and_trigger_expected_args = [
        call(42, [(10, 1), (11, 2)]),
        call(42, [(12, 3), (13, 4)]),
        call(42, [(14, 5), (15, 6)]),
    ]
    assert test_inform_selector_and_trigger.call_count == 3
    assert test_inform_selector_and_trigger.call_args_list == inform_selector_and_trigger_expected_args

    run_training_expected_args = [call(0), call(1), call(2)]
    assert test__run_training.call_count == 3
    assert test__run_training.call_args_list == run_training_expected_args

    assert test_inform_selector.call_count == 1
    test_inform_selector.assert_called_once_with(42, [(16, 7)])


@patch.object(PipelineExecutor, "_run_training")
@patch.object(GRPCHandler, "inform_selector_and_trigger")
@patch.object(GRPCHandler, "inform_selector")
@patch.object(GRPCHandler, "get_number_of_samples")
def test__handle_triggers_within_batch_empty_triggers(
    test_get_number_of_samples: MagicMock,
    test_inform_selector: MagicMock,
    test_inform_selector_and_trigger: MagicMock,
    test__run_training: MagicMock,
):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.pipeline_id = 42
    batch = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (15, 6), (16, 7)]
    triggering_indices = [-1, -1, 3]
    trigger_ids = [(0, {}), (1, {}), (2, {})]
    test_inform_selector_and_trigger.side_effect = trigger_ids
    test_inform_selector.return_value = {}
    test_get_number_of_samples.side_effect = [0, 0, 4]

    pe._handle_triggers_within_batch(batch, triggering_indices)

    inform_selector_and_trigger_expected_args = [
        call(42, []),
        call(42, []),
        call(42, [(10, 1), (11, 2), (12, 3), (13, 4)]),
    ]
    assert test_inform_selector_and_trigger.call_count == 3
    assert test_inform_selector_and_trigger.call_args_list == inform_selector_and_trigger_expected_args

    run_training_expected_args = [call(2)]
    assert test__run_training.call_count == 1
    assert test__run_training.call_args_list == run_training_expected_args

    assert test_inform_selector.call_count == 1
    test_inform_selector.assert_called_once_with(42, [(14, 5), (15, 6), (16, 7)])


@patch.object(GRPCHandler, "store_trained_model", return_value=101)
@patch.object(GRPCHandler, "start_training", return_value=1337)
@patch.object(GRPCHandler, "start_evaluation")
@patch.object(GRPCHandler, "wait_for_training_completion")
def test_run_training(
    test_wait_for_training_completion: MagicMock,
    test_start_evaluation: MagicMock,
    test_start_training: MagicMock,
    test_store_trained_model: MagicMock,
):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.pipeline_id = 42
    pe.trigger_infos = {21: {}}
    pe._run_training(21)
    assert pe.previous_model_id == 101
    assert pe.current_training_id == 1337

    test_wait_for_training_completion.assert_called_once_with(1337, 42, 21)
    test_start_training.assert_called_once_with(42, 21, get_minimal_pipeline_config(), None)
    test_store_trained_model.assert_called_once()
    test_start_evaluation.assert_not_called()


@patch.object(GRPCHandler, "store_trained_model", return_value=101)
@patch.object(GRPCHandler, "start_training", return_value=1337)
@patch.object(GRPCHandler, "store_evaluation_results")
@patch.object(GRPCHandler, "wait_for_evaluation_completion")
@patch.object(GRPCHandler, "start_evaluation")
@patch.object(GRPCHandler, "wait_for_training_completion")
def test_run_training_with_evaluation(
    test_wait_for_training_completion: MagicMock,
    test_start_evaluation: MagicMock,
    test_wait_for_evaluation_completion: MagicMock,
    test_store_evaluation_results: MagicMock,
    test_start_training: MagicMock,
    test_store_trained_model: MagicMock,
):
    evaluations = {1: EvaluationStatusReporter(TRAINING_STATUS_QUEUE, EVAL_ID, "MNIST_eval", 1000)}
    test_start_evaluation.return_value = evaluations
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    evaluation_pipeline_config = get_minimal_pipeline_config()
    evaluation_pipeline_config["evaluation"] = get_minimal_evaluation_config()
    evaluation_pipeline_config["evaluation"]["result_writers"] = ["json"]
    pe.pipeline_config = evaluation_pipeline_config

    pe.pipeline_id = 42
    pe.trigger_infos = {21: {}}
    pe._run_training(21)
    assert pe.previous_model_id == 101
    assert pe.current_training_id == 1337

    test_wait_for_training_completion.assert_called_once_with(1337, 42, 21)
    test_start_training.assert_called_once_with(42, 21, evaluation_pipeline_config, None)
    test_store_trained_model.assert_called_once()

    test_start_evaluation.assert_called_once_with(101, evaluation_pipeline_config)
    test_wait_for_evaluation_completion.assert_called_once_with(1337, evaluations)
    test_store_evaluation_results.assert_called_once()
    assert len(test_store_evaluation_results.call_args[0][0]) == 1
    result_writer: AbstractEvaluationResultWriter = test_store_evaluation_results.call_args[0][0][0]
    assert result_writer.eval_directory == EVALUATION_DIRECTORY
    assert result_writer.pipeline_id == 42
    assert result_writer.trigger_id == 21


@patch.object(GRPCHandler, "get_data_in_interval", return_value=[([(10, 1), (11, 2)], 0)])
@patch.object(PipelineExecutor, "_handle_new_data")
def test_replay_data_closed_interval(test__handle_new_data: MagicMock, test_get_data_in_interval: MagicMock):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.start_replay_at = 0
    pe.stop_replay_at = 42
    pe.replay_data()

    test_get_data_in_interval.assert_called_once_with("test", 0, 42)
    test__handle_new_data.assert_called_once_with([(10, 1), (11, 2)])


@patch.object(GRPCHandler, "get_data_in_interval", return_value=[([(10, 1)], 0), ([(11, 2)], 0)])
@patch.object(PipelineExecutor, "_handle_new_data")
def test_replay_data_closed_interval_batched(test__handle_new_data: MagicMock, test_get_data_in_interval: MagicMock):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.start_replay_at = 0
    pe.stop_replay_at = 42
    pe.replay_data()

    test_get_data_in_interval.assert_called_once_with("test", 0, 42)
    assert test__handle_new_data.call_count == 2
    assert test__handle_new_data.call_args_list == [call([(10, 1)]), call([(11, 2)])]


@patch.object(GRPCHandler, "get_new_data_since", return_value=[([(10, 1), (11, 2)], 0)])
@patch.object(PipelineExecutor, "_handle_new_data")
def test_replay_data_open_interval(test__handle_new_data: MagicMock, test_get_new_data_since: MagicMock):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.start_replay_at = 0
    pe.stop_replay_at = None
    pe.replay_data()

    test_get_new_data_since.assert_called_once_with("test", 0)
    test__handle_new_data.assert_called_once_with([(10, 1), (11, 2)])


@patch.object(GRPCHandler, "get_new_data_since", return_value=[([(10, 1)], 0), ([(11, 2)], 0)])
@patch.object(PipelineExecutor, "_handle_new_data")
def test_replay_data_open_interval_batched(test__handle_new_data: MagicMock, test_get_new_data_since: MagicMock):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.start_replay_at = 0
    pe.stop_replay_at = None
    pe.replay_data()

    test_get_new_data_since.assert_called_once_with("test", 0)
    assert test__handle_new_data.call_count == 2
    assert test__handle_new_data.call_args_list == [call([(10, 1)]), call([(11, 2)])]


@patch.object(PipelineExecutor, "init_cluster_connection", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(PipelineExecutor, "get_dataset_selector_batch_size")
@patch.object(PipelineExecutor, "replay_data")
@patch.object(PipelineExecutor, "wait_for_new_data")
def test_non_experiment_pipeline(
    test_wait_for_new_data: MagicMock,
    test_replay_data: MagicMock,
    test_get_dataset_selector_batch_size: MagicMock,
    test_grpc_connection_established,
    test_init_cluster_connection,
):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.experiment_mode = False
    pe.init_cluster_connection()
    pe.execute()

    test_get_dataset_selector_batch_size.assert_called_once()
    test_wait_for_new_data.assert_called_once_with(21)
    test_replay_data.assert_not_called()


@patch.object(PipelineExecutor, "init_cluster_connection", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(PipelineExecutor, "get_dataset_selector_batch_size")
@patch.object(PipelineExecutor, "replay_data")
@patch.object(PipelineExecutor, "wait_for_new_data")
def test_experiment_pipeline(
    test_wait_for_new_data: MagicMock,
    test_replay_data: MagicMock,
    test_get_dataset_selector_batch_size: MagicMock,
    test_grpc_connection_established,
    test_init_cluster_connection,
):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.experiment_mode = True
    pe.init_cluster_connection()
    pe.execute()

    test_get_dataset_selector_batch_size.assert_called_once()
    test_wait_for_new_data.assert_not_called()
    test_replay_data.assert_called_once()


@patch.object(PipelineExecutor, "init_cluster_connection", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(PipelineExecutor, "execute", return_value=None)
def test_execute_pipeline(
    test_execute: MagicMock,
    test_grpc_connection_established: MagicMock,
    test_init_cluster_connection: MagicMock,
):
    execute_pipeline(
        START_TIMESTAMP,
        PIPELINE_ID,
        get_minimal_system_config(),
        get_minimal_pipeline_config(),
        EVALUATION_DIRECTORY,
        SUPPORTED_EVAL_RESULT_WRITERS,
        EXCEPTION_QUEUE,
        PIPELINE_STATUS_QUEUE,
        TRAINING_STATUS_QUEUE,
        EVAL_STATUS_QUEUE,
    )

    test_init_cluster_connection.assert_called_once()
    test_execute.assert_called_once()


@patch.object(GRPCHandler, "wait_for_evaluation_completion")
@patch.object(GRPCHandler, "store_evaluation_results")
def test_build_evaluation_matrix_eval_failure(
    test_store_evaluation_results,
    test_wait_for_evaluation_completion,
):
    pipeline_config = get_minimal_pipeline_config()
    evaluation_config = get_minimal_evaluation_config()
    pipeline_config["evaluation"] = evaluation_config
    eval_dataset_config = evaluation_config["datasets"][0]
    evaluation_config["matrix_eval_dataset_id"] = eval_dataset_config["dataset_id"]
    evaluation_config["eval_strategy"] = get_minimal_eval_strategies_config()
    evaluator_stub_mock = mock.Mock(spec=["evaluate_model"])
    success_response = EvaluateModelResponse(evaluation_started=True, evaluation_id=42, dataset_size=10)
    failure_response = EvaluateModelResponse(
        evaluation_started=False, not_start_reason=EvaluationNotStartReason.EMPTY_DATASET
    )
    # the failure response shouldn't affect the following evaluations
    evaluator_stub_mock.evaluate_model.side_effect = [success_response, failure_response, success_response]

    pipeline_executor = PipelineExecutor(
        START_TIMESTAMP,
        PIPELINE_ID,
        get_minimal_system_config(),
        pipeline_config,
        str(EVALUATION_DIRECTORY),
        SUPPORTED_EVAL_RESULT_WRITERS,
        PIPELINE_STATUS_QUEUE,
        TRAINING_STATUS_QUEUE,
        EVAL_STATUS_QUEUE,
        evaluation_matrix=True,
    )
    pipeline_executor.grpc.evaluator = evaluator_stub_mock
    pipeline_executor.triggers = [0]
    pipeline_executor.trigger_infos = {
        0: {"model_id": 0, "first_timestamp": 70, "last_timestamp": 100},
    }

    def fake(first_timestamp: int, last_timestamp: int):
        yield from [(0, 100), (100, 200), (200, 300)]

    with patch.object(PeriodicalEvalStrategy, "get_eval_interval", side_effect=fake):
        pipeline_executor.build_evaluation_matrix()
        assert evaluator_stub_mock.evaluate_model.call_count == 3
        assert test_store_evaluation_results.call_count == 2
        assert test_wait_for_evaluation_completion.call_count == 2

        assert pipeline_executor.pipeline_log["evaluation_matrix"][0][f"{0}-{100}"] != {}
        assert pipeline_executor.pipeline_log["evaluation_matrix"][0][f"{100}-{200}"] == {
            "failure_reason": "EMPTY_DATASET"
        }
        assert pipeline_executor.pipeline_log["evaluation_matrix"][0][f"{200}-{300}"] != {}


@patch.object(GRPCHandler, "wait_for_evaluation_completion")
@patch.object(GRPCHandler, "store_evaluation_results")
def test_build_evaluation_matrix(
    test_store_evaluation_results,
    test_wait_for_evaluation_completion,
):
    pipeline_config = get_minimal_pipeline_config()
    evaluation_config = get_minimal_evaluation_config()
    pipeline_config["evaluation"] = evaluation_config
    eval_dataset_config = evaluation_config["datasets"][0]
    evaluation_config["matrix_eval_dataset_id"] = eval_dataset_config["dataset_id"]
    evaluation_config["eval_strategy"] = get_minimal_eval_strategies_config()
    evaluator_stub_mock = mock.Mock(spec=["evaluate_model"])
    evaluator_stub_mock.evaluate_model.return_value = EvaluateModelResponse(
        evaluation_started=True, evaluation_id=42, dataset_size=10
    )

    pipeline_executor = PipelineExecutor(
        START_TIMESTAMP,
        PIPELINE_ID,
        get_minimal_system_config(),
        pipeline_config,
        str(EVALUATION_DIRECTORY),
        SUPPORTED_EVAL_RESULT_WRITERS,
        PIPELINE_STATUS_QUEUE,
        TRAINING_STATUS_QUEUE,
        EVAL_STATUS_QUEUE,
        evaluation_matrix=True,
    )
    pipeline_executor.grpc.evaluator = evaluator_stub_mock
    pipeline_executor.triggers = [0, 1]
    pipeline_executor.trigger_infos = {
        0: {"model_id": 0, "first_timestamp": 79, "last_timestamp": 100},
        1: {"model_id": 1, "first_timestamp": 132, "last_timestamp": 200},
    }

    def fake(first_timestamp: int, last_timestamp: int):
        yield from [(0, 100), (100, 200), (200, 300)]

    with (
        patch.object(
            GRPCHandler, "prepare_evaluation_request", wraps=pipeline_executor.grpc.prepare_evaluation_request
        ) as prepare_evaluation_request_wrap,
        patch.object(PeriodicalEvalStrategy, "get_eval_interval", side_effect=fake),
    ):
        pipeline_executor.build_evaluation_matrix()
        assert evaluator_stub_mock.evaluate_model.call_count == 6

        for trigger_id in pipeline_executor.triggers:
            expected_model_id = pipeline_executor.trigger_infos[trigger_id]["model_id"]
            prepare_evaluation_request_wrap.assert_any_call(
                eval_dataset_config, expected_model_id, evaluation_config["device"], 0, 100
            )
            prepare_evaluation_request_wrap.assert_any_call(
                eval_dataset_config, expected_model_id, evaluation_config["device"], 100, 200
            )
            prepare_evaluation_request_wrap.assert_any_call(
                eval_dataset_config, expected_model_id, evaluation_config["device"], 200, 300
            )
