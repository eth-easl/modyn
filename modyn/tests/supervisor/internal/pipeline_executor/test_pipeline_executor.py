# pylint: disable=unused-argument,redefined-outer-name
import multiprocessing as mp
import os
import pathlib
import shutil
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock, call, patch

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluateModelResponse, EvaluationAbortedReason
from modyn.supervisor.internal.eval_strategies.matrix_eval_strategy import MatrixEvalStrategy
from modyn.supervisor.internal.evaluation_result_writer import JsonResultWriter, TensorboardResultWriter
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.pipeline_executor import EVALUATION_RESULTS, PipelineExecutor, execute_pipeline

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
        "dataset": {
            "dataset_id": "MNIST_eval",
            "bytes_parser_function": "def bytes_parser_function(data: bytes) -> bytes:\n\treturn data",
            "dataloader_workers": 2,
            "batch_size": 64,
            "metrics": [{"name": "Accuracy"}],
        },
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
        "name": "MatrixEvalStrategy",
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


def get_non_connecting_pipeline_executor(pipeline_config=None) -> PipelineExecutor:
    if pipeline_config is None:
        pipeline_config = get_minimal_pipeline_config()
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
    # each tuple is (key, timestamp, label)
    batch = [(10, 1, 5), (11, 2, 5), (12, 3, 5), (13, 4, 5), (14, 5, 5), (15, 6, 5), (16, 7, 5)]
    triggering_indices = [-1, -1, 3]
    trigger_ids = [(0, {}), (1, {}), (2, {})]
    test_inform_selector_and_trigger.side_effect = trigger_ids
    test_inform_selector.return_value = {}
    test_get_number_of_samples.side_effect = [0, 0, 4]
    pe._handle_triggers_within_batch(batch, triggering_indices)
    inform_selector_and_trigger_expected_args = [
        call(42, []),
        call(42, []),
        call(42, [(10, 1, 5), (11, 2, 5), (12, 3, 5), (13, 4, 5)]),
    ]
    assert test_inform_selector_and_trigger.call_count == 3
    assert test_inform_selector_and_trigger.call_args_list == inform_selector_and_trigger_expected_args

    run_training_expected_args = [call(2)]
    assert test__run_training.call_count == 1
    assert test__run_training.call_args_list == run_training_expected_args

    assert test_inform_selector.call_count == 1
    test_inform_selector.assert_called_once_with(42, [(14, 5, 5), (15, 6, 5), (16, 7, 5)])


@patch.object(PipelineExecutor, "_run_training")
@patch.object(GRPCHandler, "inform_selector_and_trigger")
@patch.object(GRPCHandler, "inform_selector")
@patch.object(GRPCHandler, "get_number_of_samples")
def test__handle_triggers_within_batch_trigger_timespan(
    test_get_number_of_samples: MagicMock,
    test_inform_selector: MagicMock,
    test_inform_selector_and_trigger: MagicMock,
    test__run_training: MagicMock,
):
    def reset_state():
        test_get_number_of_samples.reset_mock()
        test_inform_selector_and_trigger.reset_mock()
        test__run_training.reset_mock()
        test_inform_selector.reset_mock()

    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    test_inform_selector.return_value = {}
    test_inform_selector_and_trigger.side_effect = [(0, {}), (1, {}), (2, {}), (3, {}), (4, {}), (5, {})]

    # each tuple is (key, timestamp, label)
    first_batch = [
        # trigger 0
        (10, 1, 5),
        (11, 2, 5),
        # trigger 1 is empty
        # trigger 2
        (12, 3, 5),
        (13, 4, 5),
        # remaining data
        (14, 5, 5),
        (15, 6, 5),
        (16, 7, 5),
    ]
    triggering_indices = [1, 1, 3]
    test_get_number_of_samples.side_effect = [2, 0, 2]  # the size of trigger 0, 1, 2

    pe._handle_triggers_within_batch(first_batch, triggering_indices)
    assert pe.pipeline_log["supervisor"]["triggers"][0]["first_timestamp"] == 1
    assert pe.pipeline_log["supervisor"]["triggers"][0]["last_timestamp"] == 2

    assert "first_timestamp" not in pe.pipeline_log["supervisor"]["triggers"][1]
    assert "last_timestamp" not in pe.pipeline_log["supervisor"]["triggers"][1]

    assert pe.pipeline_log["supervisor"]["triggers"][2]["first_timestamp"] == 3
    assert pe.pipeline_log["supervisor"]["triggers"][2]["last_timestamp"] == 4
    assert pe.remaining_data_range == (5, 7)
    reset_state()

    second_batch = [
        # trigger 3 covers remaining data from last batch
        # trigger 4
        (17, 8, 5),
        (18, 9, 5),
        # remaining data
        (19, 10, 5),
    ]
    triggering_indices = [-1, 1]
    test_get_number_of_samples.side_effect = [3, 2]  # the size of trigger 3, 4
    pe._handle_triggers_within_batch(second_batch, triggering_indices)
    assert pe.pipeline_log["supervisor"]["triggers"][3]["first_timestamp"] == 5
    assert pe.pipeline_log["supervisor"]["triggers"][3]["last_timestamp"] == 7
    assert pe.pipeline_log["supervisor"]["triggers"][4]["first_timestamp"] == 8
    assert pe.pipeline_log["supervisor"]["triggers"][4]["last_timestamp"] == 9
    assert pe.remaining_data_range == (10, 10)
    reset_state()

    third_batch = [(20, 11, 5), (21, 12, 5), (22, 13, 5)]  # trigger 5, also includes remaining data from last batch
    triggering_indices = [2]
    test_get_number_of_samples.side_effect = [4]  # the size of trigger 5
    pe._handle_triggers_within_batch(third_batch, triggering_indices)
    assert pe.pipeline_log["supervisor"]["triggers"][5]["first_timestamp"] == 10
    assert pe.pipeline_log["supervisor"]["triggers"][5]["last_timestamp"] == 13
    assert pe.remaining_data_range is None


@patch.object(PipelineExecutor, "_run_training")
@patch.object(GRPCHandler, "inform_selector_and_trigger")
@patch.object(GRPCHandler, "inform_selector")
@patch.object(GRPCHandler, "get_number_of_samples")
def test__handle_triggers_within_batch_trigger_timespan_across_batch(
    test_get_number_of_samples: MagicMock,
    test_inform_selector: MagicMock,
    test_inform_selector_and_trigger: MagicMock,
    test__run_training: MagicMock,
):
    def reset_state():
        test_get_number_of_samples.reset_mock()
        test_inform_selector_and_trigger.reset_mock()
        test__run_training.reset_mock()
        test_inform_selector.reset_mock()

    pe = get_non_connecting_pipeline_executor()
    test_inform_selector.return_value = {}
    test_inform_selector_and_trigger.side_effect = [(0, {}), (1, {}), (2, {})]

    # each tuple is (key, timestamp, label)
    first_batch = [
        # trigger 0
        (10, 1, 5),
        (11, 2, 5),
    ]
    triggering_indices = []
    pe._handle_triggers_within_batch(first_batch, triggering_indices)
    assert pe.remaining_data_range == (1, 2)
    test_get_number_of_samples.assert_not_called()
    test_inform_selector_and_trigger.assert_not_called()
    test__run_training.assert_not_called()
    test_inform_selector.assert_called_once_with(PIPELINE_ID, first_batch)

    reset_state()
    second_batch = [
        # trigger 1
        (12, 3, 5),
        (13, 4, 5),
    ]
    triggering_indices = [-1]
    test_get_number_of_samples.side_effect = [2]  # the size of trigger 0
    pe._handle_triggers_within_batch(second_batch, triggering_indices)
    assert pe.pipeline_log["supervisor"]["triggers"][0]["first_timestamp"] == 1
    assert pe.pipeline_log["supervisor"]["triggers"][0]["last_timestamp"] == 2
    assert pe.remaining_data_range == (3, 4)
    test_get_number_of_samples.assert_called_once_with(PIPELINE_ID, 0)
    test_inform_selector_and_trigger.assert_called_once_with(42, [])
    test__run_training.assert_called_once_with(0)
    test_inform_selector.assert_called_once_with(PIPELINE_ID, second_batch)

    reset_state()
    third_batch = [
        # still trigger 1's data
        (14, 5, 5),
        (15, 6, 5),
        (16, 7, 5),
    ]
    triggering_indices = []
    pe._handle_triggers_within_batch(third_batch, triggering_indices)
    assert pe.remaining_data_range == (3, 7)
    reset_state()

    fourth_batch = [
        # still trigger 1's data
        (17, 8, 5),
        (18, 9, 5),
        # trigger 2
        (19, 10, 5),
        (20, 11, 5),
    ]

    triggering_indices = [1, 3]
    test_get_number_of_samples.side_effect = [7, 2]
    pe._handle_triggers_within_batch(fourth_batch, triggering_indices)
    assert pe.remaining_data_range is None
    assert pe.pipeline_log["supervisor"]["triggers"][1]["first_timestamp"] == 3
    assert pe.pipeline_log["supervisor"]["triggers"][1]["last_timestamp"] == 9
    assert pe.pipeline_log["supervisor"]["triggers"][2]["first_timestamp"] == 10
    assert pe.pipeline_log["supervisor"]["triggers"][2]["last_timestamp"] == 11


@patch.object(GRPCHandler, "store_trained_model", return_value=101)
@patch.object(GRPCHandler, "start_training", return_value=1337)
@patch.object(GRPCHandler, "wait_for_training_completion")
def test__run_training(
    test_wait_for_training_completion: MagicMock,
    test_start_training: MagicMock,
    test_store_trained_model: MagicMock,
):
    pe = get_non_connecting_pipeline_executor()  # pylint: disable=no-value-for-parameter
    pe.pipeline_id = 42
    pe._run_training(21)
    assert pe.previous_model_id == 101
    assert pe.current_training_id == 1337

    test_wait_for_training_completion.assert_called_once_with(1337, 42, 21)
    test_start_training.assert_called_once_with(42, 21, get_minimal_pipeline_config(), None, None)
    test_store_trained_model.assert_called_once()


@patch.object(GRPCHandler, "store_trained_model", return_value=101)
@patch.object(GRPCHandler, "start_training", return_value=1337)
@patch.object(GRPCHandler, "wait_for_training_completion")
def test__run_training_set_num_samples_to_pass(
    test_wait_for_training_completion: MagicMock,
    test_start_training: MagicMock,
    test_store_trained_model: MagicMock,
):
    pipeline_config = get_minimal_pipeline_config()
    pipeline_config["training"]["num_samples_to_pass"] = [73]
    pe = get_non_connecting_pipeline_executor(pipeline_config)
    pe.pipeline_id = 42
    pe._run_training(trigger_id=21)
    test_start_training.assert_called_once_with(42, 21, pipeline_config, None, 73)
    test_start_training.reset_mock()

    # the next time _run_training is called, the num_samples_to_pass should be set to 0
    # because the next trigger is out of the range of `num_samples_to_pass`
    pe._run_training(trigger_id=22)
    test_start_training.assert_called_once_with(42, 22, pipeline_config, 101, None)


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
def test__start_evaluations_success(
    test_store_evaluation_results,
    test_wait_for_evaluation_completion,
):
    pipeline_config = get_minimal_pipeline_config()
    evaluation_config = get_minimal_evaluation_config()
    eval_dataset_config = evaluation_config["dataset"]
    pipeline_config["evaluation"] = evaluation_config
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
    )
    pipeline_executor.grpc.evaluator = evaluator_stub_mock
    pipeline_executor.current_training_id = 0

    def fake(first_timestamp: int, last_timestamp: int):
        yield from [(0, 100), (100, 200), (200, 300)]

    with patch.object(MatrixEvalStrategy, "get_eval_intervals", side_effect=fake):
        model_id = 1
        pipeline_executor._start_evaluations(
            trigger_id=0, model_id=model_id, trigger_set_first_timestamp=20, trigger_set_last_timestamp=70
        )
        device = evaluation_config["device"]
        expected_calls = [
            call(GRPCHandler.prepare_evaluation_request(eval_dataset_config, model_id, device, 0, 100)),
            call(GRPCHandler.prepare_evaluation_request(eval_dataset_config, model_id, device, 100, 200)),
            call(GRPCHandler.prepare_evaluation_request(eval_dataset_config, model_id, device, 200, 300)),
        ]
        assert evaluator_stub_mock.evaluate_model.call_count == 3
        assert evaluator_stub_mock.evaluate_model.call_args_list == expected_calls


@patch.object(GRPCHandler, "wait_for_evaluation_completion")
@patch.object(GRPCHandler, "store_evaluation_results")
def test__start_evaluations_failure(
        test_store_evaluation_results,
        test_wait_for_evaluation_completion,
):
    pipeline_config = get_minimal_pipeline_config()
    evaluation_config = get_minimal_evaluation_config()
    pipeline_config["evaluation"] = evaluation_config
    evaluation_config["eval_strategy"] = get_minimal_eval_strategies_config()
    evaluator_stub_mock = mock.Mock(spec=["evaluate_model"])
    success_response = EvaluateModelResponse(evaluation_started=True, evaluation_id=42, dataset_size=10)
    failure_response = EvaluateModelResponse(
        evaluation_started=False, eval_aborted_reason=EvaluationAbortedReason.EMPTY_DATASET
    )
    # we let the second evaluation fail; it shouldn't affect the third evaluation
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
    )
    pipeline_executor.grpc.evaluator = evaluator_stub_mock
    pipeline_executor.current_training_id = 0

    def fake_get_eval_intervals(first_timestamp: int, last_timestamp: int):
        yield from [(0, 100), (100, 200), (200, 300)]

    with patch.object(MatrixEvalStrategy, "get_eval_intervals", side_effect=fake_get_eval_intervals):
        pipeline_executor._start_evaluations(
            trigger_id=0, model_id=0, trigger_set_first_timestamp=70, trigger_set_last_timestamp=100
        )
        assert evaluator_stub_mock.evaluate_model.call_count == 3
        assert test_store_evaluation_results.call_count == 2
        assert test_wait_for_evaluation_completion.call_count == 2

        assert pipeline_executor.pipeline_log[EVALUATION_RESULTS][0][f"{0}-{100}"] != {}
        assert pipeline_executor.pipeline_log[EVALUATION_RESULTS][0][f"{100}-{200}"] == {
            "failure_reason": "EMPTY_DATASET"
        }
        assert pipeline_executor.pipeline_log[EVALUATION_RESULTS][0][f"{200}-{300}"] != {}
