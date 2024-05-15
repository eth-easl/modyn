# pylint: disable=unused-argument,redefined-outer-name
import multiprocessing as mp
import os
import pathlib
import shutil
from dataclasses import dataclass
from typing import overload
from unittest.mock import ANY, MagicMock, PropertyMock, call, patch

import pytest
from modyn.config.schema.config import DatasetsConfig, ModynConfig, SupervisorConfig
from modyn.config.schema.pipeline import EvaluationConfig, ModynPipelineConfig
from modyn.supervisor.internal.evaluation_result_writer import (
    AbstractEvaluationResultWriter,
    JsonResultWriter,
    TensorboardResultWriter,
)
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.pipeline_executor import PipelineExecutor, execute_pipeline
from modyn.supervisor.internal.pipeline_executor.models import (
    ConfigLogs,
    ExecutionState,
    PipelineLogs,
    PipelineOptions,
    StageLog,
)
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


@pytest.fixture
def minimal_system_config(dummy_system_config: ModynConfig) -> ModynConfig:
    config = dummy_system_config.model_copy()
    config.supervisor = SupervisorConfig(hostname="localhost", port=50051, eval_directory=EVALUATION_DIRECTORY)
    return config


@pytest.fixture
def minimal_pipeline_config(dummy_pipeline_config: ModynPipelineConfig) -> ModynPipelineConfig:
    config = dummy_pipeline_config.model_copy()
    return config


def get_dummy_pipeline_options(system_config: ModynConfig, pipeline_config: ModynPipelineConfig) -> PipelineOptions:
    return PipelineOptions(
        start_timestamp=START_TIMESTAMP,
        pipeline_id=PIPELINE_ID,
        modyn_config=system_config,
        pipeline_config=pipeline_config,
        eval_directory=str(EVALUATION_DIRECTORY),
        supervisor_supported_eval_result_writers=SUPPORTED_EVAL_RESULT_WRITERS,
        exception_queue=EXCEPTION_QUEUE,
        pipeline_status_queue=PIPELINE_STATUS_QUEUE,
        training_status_queue=TRAINING_STATUS_QUEUE,
        eval_status_queue=EVAL_STATUS_QUEUE,
    )


@pytest.fixture
def dummy_pipeline_options(
    minimal_system_config: ModynConfig, minimal_pipeline_config: ModynPipelineConfig
) -> PipelineOptions:
    return get_dummy_pipeline_options(minimal_system_config, minimal_pipeline_config)


@pytest.fixture
def dummy_execution_state(dummy_pipeline_options: PipelineOptions) -> ExecutionState:
    return ExecutionState(**vars(dummy_pipeline_options))


@pytest.fixture
def dummy_logs(dummy_pipeline_options: PipelineOptions) -> PipelineLogs:
    options = dummy_pipeline_options
    return PipelineLogs(
        pipeline_id=PIPELINE_ID,
        config=ConfigLogs(system=options.modyn_config, pipeline=options.pipeline_config),
        experiment=options.experiment_mode,
        start_replay_at=options.start_replay_at,
        stop_replay_at=options.stop_replay_at,
    )


@pytest.fixture
def dummy_stage_log() -> StageLog:
    return StageLog(id="dummy", start=0, sample_idx=1, sample_time=1000)


@overload
def get_non_connecting_pipeline_executor(
    system_config: ModynConfig, pipeline_config: ModynPipelineConfig
) -> PipelineExecutor: ...


@overload
def get_non_connecting_pipeline_executor(pipeline_options: PipelineOptions) -> PipelineExecutor: ...


def get_non_connecting_pipeline_executor(
    pipeline_options: PipelineOptions | None = None,
    system_config: ModynConfig | None = None,
    pipeline_config: ModynPipelineConfig | None = None,
) -> PipelineExecutor:
    if pipeline_options:
        return PipelineExecutor(pipeline_options)
    return PipelineExecutor(get_dummy_pipeline_options(system_config, pipeline_config))


@pytest.fixture
def non_connecting_pipeline_executor(dummy_pipeline_options: PipelineOptions) -> PipelineExecutor:
    return PipelineExecutor(dummy_pipeline_options)


def sleep_mock(duration: int):
    raise KeyboardInterrupt


def setup():
    if EVALUATION_DIRECTORY.is_dir():
        shutil.rmtree(EVALUATION_DIRECTORY)
    EVALUATION_DIRECTORY.mkdir(0o777)


def teardown():
    shutil.rmtree(EVALUATION_DIRECTORY)


def test_initialization(non_connecting_pipeline_executor: PipelineExecutor) -> None:
    assert non_connecting_pipeline_executor.state.stage == PipelineStage.INIT


def test_get_dataset_selector_batch_size_given(
    minimal_system_config: ModynConfig,
    minimal_pipeline_config: ModynPipelineConfig,
    dummy_dataset_config: DatasetsConfig,
):
    dataset1 = dummy_dataset_config.model_copy()
    dataset1.selector_batch_size = 2048
    minimal_system_config.storage.datasets = [dataset1]
    pe = get_non_connecting_pipeline_executor(
        system_config=minimal_system_config, pipeline_config=minimal_pipeline_config
    )
    pe.state.selector_batch_size == 2048


def test_shutdown_trainer():
    # TODO(MaxiBoether): implement
    pass


@patch.object(GRPCHandler, "get_new_data_since", return_value=[([(10, 42, 0), (11, 43, 1)], 99)])
@patch.object(PipelineExecutor, "_process_new_data", side_effect=[[10, 11]])
def test_fetch_new_data(
    test__process_new_data: MagicMock,
    test_get_new_data_since: MagicMock,
    non_connecting_pipeline_executor: PipelineExecutor,
    dummy_execution_state: ExecutionState,
    dummy_logs: PipelineLogs,
):
    dummy_execution_state.max_timestamp = 21
    triggers = non_connecting_pipeline_executor._fetch_new_data(dummy_execution_state, dummy_logs)

    test_get_new_data_since.assert_called_once_with("test", 21)
    test__process_new_data.assert_called_once_with(ANY, ANY, [(10, 42, 0), (11, 43, 1)], 99)
    assert triggers == 2


@patch.object(GRPCHandler, "get_new_data_since", return_value=[([(10, 42, 0)], 98), ([(11, 43, 1)], 99)])
@patch.object(PipelineExecutor, "_process_new_data", side_effect=[[10], [11]])
def test_fetch_new_data_batched(
    test__process_new_data: MagicMock,
    test_get_new_data_since: MagicMock,
    non_connecting_pipeline_executor: PipelineExecutor,
    dummy_execution_state: ExecutionState,
    dummy_logs: PipelineLogs,
):
    dummy_execution_state.max_timestamp = 21
    triggers = non_connecting_pipeline_executor._fetch_new_data(dummy_execution_state, dummy_logs)
    test_get_new_data_since.assert_called_once_with("test", 21)

    expected_calls = [call(ANY, ANY, [(10, 42, 0)], 98), call(ANY, ANY, [(11, 43, 1)], 99)]
    assert test__process_new_data.call_args_list == expected_calls
    assert triggers == 2


def test_serve_online_data(
    non_connecting_pipeline_executor: PipelineExecutor, dummy_execution_state: ExecutionState, dummy_logs: PipelineLogs
):
    pe = non_connecting_pipeline_executor

    mocked__process_new_data_return_vals = [[10], [], []]
    mocked_get_new_data_since = [
        [([(10, 42, 0), (11, 43, 0), (12, 43, 1)], 97)],
        [([(11, 43, 0), (12, 43, 1), (13, 43, 2), (14, 45, 3)], 98)],
        [([], 99)],
        KeyboardInterrupt,
    ]

    handle_mock: MagicMock
    with patch.object(pe, "_process_new_data", side_effect=mocked__process_new_data_return_vals) as handle_mock:
        get_new_data_mock: MagicMock
        with patch.object(pe.grpc, "get_new_data_since", side_effect=mocked_get_new_data_since) as get_new_data_mock:
            dummy_execution_state.max_timestamp = 21
            pe._serve_online_data(dummy_execution_state, dummy_logs)

            assert handle_mock.call_count == 3
            assert get_new_data_mock.call_count == 3 + 1

            expected_handle_mock_arg_list = [
                call(ANY, ANY, [(10, 42, 0), (11, 43, 0), (12, 43, 1)], 97),
                call(ANY, ANY, [(13, 43, 2), (14, 45, 3)], 98),
                call(ANY, ANY, [], 99),
            ]
            assert handle_mock.call_args_list == expected_handle_mock_arg_list

            expected_get_new_data_arg_list = [call("test", 21), call("test", 43), call("test", 45), call("test", 45)]
            assert get_new_data_mock.call_args_list == expected_get_new_data_arg_list


@pytest.mark.parametrize(
    "selector_batch_size, batching_return_vals, expected_process_new_data_batch_args",
    [
        (
            3,
            [[10], [14, 15], []],
            [
                call(ANY, ANY, [(10, 1), (11, 2), (12, 3)]),
                call(ANY, ANY, [(13, 4), (14, 5), (15, 6)]),
                call(ANY, ANY, [(16, 7), (17, 8)]),
            ],
        ),
        (200, [[10, 14, 15]], None),  # large batch  # use new_data
        (200, [[]], None),  # no triggers  # use new_data
    ],
)
def test__process_new_data(
    selector_batch_size: int,
    batching_return_vals: list[list[int]],
    expected_process_new_data_batch_args: list,
    non_connecting_pipeline_executor: PipelineExecutor,
    dummy_execution_state: ExecutionState,
):
    pe = non_connecting_pipeline_executor
    new_data = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (15, 6), (16, 7), (17, 8)]

    batch_mock: MagicMock
    with patch.object(ExecutionState, "selector_batch_size", new_callable=PropertyMock) as selector_batch_size_mock:
        selector_batch_size_mock.return_value = selector_batch_size
        with patch.object(PipelineExecutor, "_process_new_data_batch", side_effect=batching_return_vals) as batch_mock:
            trigger_indexes = pe._process_new_data(dummy_execution_state, pe.logs, new_data, 0)
            assert trigger_indexes == [x for sub in batching_return_vals for x in sub]
            assert batch_mock.call_args_list == (expected_process_new_data_batch_args or [call(ANY, ANY, new_data)])


@patch.object(GRPCHandler, "inform_selector", return_value={})
def test__process_new_data_batch_no_triggers(test_inform_selector: MagicMock, dummy_pipeline_options: PipelineOptions):
    dummy_pipeline_options.pipeline_id = 42
    pe = get_non_connecting_pipeline_executor(dummy_pipeline_options)
    batch = [(10, 1), (11, 2)]

    with patch.object(pe.trigger, "inform", return_value=[]) as inform_mock:
        assert len(pe._process_new_data_batch(pe.state, pe.logs, batch)) == 0

        inform_mock.assert_called_once_with(batch)
        test_inform_selector.assert_called_once_with(42, batch)


@pytest.mark.parametrize(
    (
        "batch,trigger_indexes,trigger_ids,inform_selector_and_trigger_expected_args,"
        "train_and_store_model_expected_args,inform_selector_expected_args"
    ),
    [
        (
            [(10, 1, 0), (11, 2, 0), (12, 3, 0), (13, 4, 0), (14, 5, 0), (15, 6, 0), (16, 7, 0)],  # batch
            [1, 3, 5],  # trigger_indexes
            [(0, {}), (1, {}), (2, {})],  # trigger_id, selector_log
            [  # inform_selector_and_trigger_expected_args
                [(10, 1, 0), (11, 2, 0)],
                [(12, 3, 0), (13, 4, 0)],
                [(14, 5, 0), (15, 6, 0)],
            ],
            # train_and_store_model_expected_args
            [call(ANY, ANY, 0), call(ANY, ANY, 1), call(ANY, ANY, 2)],
            # inform_selector_expected_args
            [call(42, [(16, 7, 0)])],
        ),
        (  # test empty triggers
            [(10, 1, 5), (11, 2, 5), (12, 3, 5), (13, 4, 5), (14, 5, 5), (15, 6, 5), (16, 7, 5)],  # batch
            [0, 0, 3],  # trigger_indexes
            [(0, {}), (1, {}), (2, {})],  # (trigger_id, selector_log)
            # inform_selector_and_trigger_expected_args
            [[(10, 1, 5)], [], [(11, 2, 5), (12, 3, 5), (13, 4, 5)]],
            # train_and_store_model_expected_args
            [call(ANY, ANY, 0), call(ANY, ANY, 2)],
            # inform_selector_expected_args
            [call(42, [(14, 5, 5), (15, 6, 5), (16, 7, 5)])],
        ),
    ],
)
@patch.object(PipelineExecutor, "_train_and_store_model", return_value=(-1, -1))
@patch.object(GRPCHandler, "inform_selector_and_trigger")
@patch.object(GRPCHandler, "inform_selector")
@patch.object(GRPCHandler, "get_number_of_samples")
def test__execute_triggers(
    test_get_number_of_samples: MagicMock,
    test_inform_selector: MagicMock,
    test_inform_selector_and_trigger: MagicMock,
    test__train_and_store_model: MagicMock,
    batch: list[tuple[int, int]],
    trigger_indexes: list[int],
    trigger_ids: list[tuple[int, dict]],  # TODO: rename
    inform_selector_and_trigger_expected_args: list,
    train_and_store_model_expected_args: list,
    inform_selector_expected_args: list,
    dummy_pipeline_options: PipelineOptions,
) -> None:
    dummy_pipeline_options.pipeline_id = 42
    dummy_pipeline_options.pipeline_config.evaluation = None
    pe = get_non_connecting_pipeline_executor(dummy_pipeline_options)

    test_inform_selector_and_trigger.side_effect = trigger_ids
    test_inform_selector.return_value = {}
    test_get_number_of_samples.side_effect = [len(params) for params in inform_selector_and_trigger_expected_args]

    pe._execute_triggers(pe.state, pe.logs, batch, trigger_indexes)
    pe._inform_selector_remaining_data(pe.state, pe.logs, batch, trigger_indexes)

    assert test_inform_selector_and_trigger.call_count == len(inform_selector_and_trigger_expected_args)
    assert test_inform_selector_and_trigger.call_args_list == (
        [call(42, params) for params in inform_selector_and_trigger_expected_args]
    )

    assert test__train_and_store_model.call_count == len(train_and_store_model_expected_args)
    assert test__train_and_store_model.call_args_list == train_and_store_model_expected_args

    assert test_inform_selector.call_count == len(inform_selector_expected_args)
    assert test_inform_selector.call_args_list == inform_selector_expected_args


@dataclass
class _BatchConfig:
    data: list[tuple[int, int, int]]  # (key, timestamp, label)
    trigger_indexes: list[int]
    get_num_samples: list[int]  # sequence of return values of get_number_of_samples
    remaining_data_range: tuple[int, int] | None = None
    expected_tracking: dict[str, list[int | None]] | None = None


@pytest.mark.parametrize(
    "batches,inform_selector_and_trigger_retval",
    [
        (  # TEST 1
            [
                _BatchConfig(  # first batch
                    data=[
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
                    ],
                    trigger_indexes=[1, 1, 3],
                    get_num_samples=[2, 0, 2],
                    remaining_data_range=(5, 7),
                    expected_tracking={
                        "trigger_i": [0, 1, 2],
                        "trigger_id": [0, 1, 2],
                        "trigger_index": [1, 1, 3],
                        "first_timestamp": [1, None, 3],
                        "last_timestamp": [2, None, 4],
                    },
                ),
                _BatchConfig(  # second batch
                    data=[
                        # trigger 3 covers remaining data from last batch
                        # trigger 4
                        (17, 8, 5),
                        (18, 9, 5),
                        # remaining data
                        (19, 10, 5),
                    ],
                    trigger_indexes=[-1, 1],
                    get_num_samples=[3, 2],
                    remaining_data_range=(10, 10),
                    expected_tracking={
                        "trigger_i": [0, 1, 2, 0, 1],
                        "trigger_id": [0, 1, 2, 3, 4],
                        "trigger_index": [1, 1, 3, -1, 1],
                        "first_timestamp": [1, None, 3, 5, 8],
                        "last_timestamp": [2, None, 4, 7, 9],
                    },
                ),
                _BatchConfig(  # third batch
                    data=[(20, 11, 5), (21, 12, 5), (22, 13, 5)],
                    trigger_indexes=[2],
                    get_num_samples=[4],
                    remaining_data_range=None,
                    expected_tracking={
                        "trigger_i": [0, 1, 2, 0, 1, 0],
                        "trigger_id": [0, 1, 2, 3, 4, 5],
                        "trigger_index": [1, 1, 3, -1, 1, 2],
                        "first_timestamp": [1, None, 3, 5, 8, 10],
                        "last_timestamp": [2, None, 4, 7, 9, 13],
                    },
                ),
            ],
            # inform_selector_and_trigger_retval
            [(0, {}), (1, {}), (2, {}), (3, {}), (4, {}), (5, {})],
        ),
        (  # TEST 2
            [
                _BatchConfig(  # first batch
                    data=[
                        # trigger 0
                        (10, 1, 5),
                        (11, 2, 5),
                    ],
                    trigger_indexes=[],
                    get_num_samples=[0, 0],
                    remaining_data_range=(1, 2),
                    expected_tracking=None,  # no trigger index
                ),
                _BatchConfig(  # second batch
                    data=[
                        # trigger 1
                        (12, 3, 5),
                        (13, 4, 5),
                    ],
                    trigger_indexes=[-1],
                    get_num_samples=[2],
                    remaining_data_range=(3, 4),
                    expected_tracking={
                        "trigger_i": [0],
                        "trigger_id": [0],
                        "trigger_index": [-1],
                        "first_timestamp": [1],
                        "last_timestamp": [2],
                    },
                ),
                _BatchConfig(  # third batch
                    data=[
                        # still trigger 1's data
                        (14, 5, 5),
                        (15, 6, 5),
                        (16, 7, 5),
                    ],
                    trigger_indexes=[],
                    get_num_samples=[],
                    remaining_data_range=(3, 7),
                    expected_tracking={  # nothing new as empty trigger
                        "trigger_i": [0],
                        "trigger_id": [0],
                        "trigger_index": [-1],
                        "first_timestamp": [1],
                        "last_timestamp": [2],
                    },
                ),
                _BatchConfig(  # fourth batch
                    data=[
                        # still trigger 1's data
                        (17, 8, 5),
                        (18, 9, 5),
                        # trigger 2
                        (19, 10, 5),
                        (20, 11, 5),
                    ],
                    trigger_indexes=[1, 3],
                    get_num_samples=[7, 2],
                    remaining_data_range=None,
                    expected_tracking={
                        "trigger_i": [0, 0, 1],
                        "trigger_id": [0, 1, 2],
                        "trigger_index": [-1, 1, 3],
                        "first_timestamp": [1, 3, 10],
                        "last_timestamp": [2, 9, 11],
                    },
                ),
            ],
            # inform_selector_and_trigger_retval
            [(0, {}), (1, {}), (2, {})],
        ),
    ],
)
@patch.object(PipelineExecutor, "_train_and_store_model", return_value=(-1, -1))
@patch.object(GRPCHandler, "inform_selector_and_trigger")
@patch.object(GRPCHandler, "inform_selector")
@patch.object(GRPCHandler, "get_number_of_samples")
def test__execute_triggers_within_batch_trigger_timespan(
    test_get_number_of_samples: MagicMock,
    test_inform_selector: MagicMock,
    test_inform_selector_and_trigger: MagicMock,
    test__run_training: MagicMock,
    batches: list[_BatchConfig],
    inform_selector_and_trigger_retval: list[tuple[int, dict]],
    dummy_pipeline_options: PipelineOptions,
) -> None:
    tracking_columns = ["trigger_i", "trigger_id", "trigger_index", "first_timestamp", "last_timestamp"]
    test_inform_selector_and_trigger.side_effect = inform_selector_and_trigger_retval
    test_inform_selector.return_value = {}

    def run_batch_triggers_and_validate(pe: PipelineExecutor, batch: _BatchConfig) -> None:
        test_get_number_of_samples.reset_mock()
        test_inform_selector_and_trigger.reset_mock()
        test__run_training.reset_mock()
        test_inform_selector.reset_mock()

        test_get_number_of_samples.side_effect = batch.get_num_samples

        pe._execute_triggers(pe.state, pe.logs, batch.data, batch.trigger_indexes)
        pe._inform_selector_remaining_data(pe.state, pe.logs, batch.data, batch.trigger_indexes)

        assert pe.state.remaining_data_range == batch.remaining_data_range
        if batch.expected_tracking:
            assert pe.state.tracking[PipelineStage.EXECUTE_SINGLE_TRIGGER.name][tracking_columns].to_dict("list") == (
                batch.expected_tracking
            )

    pe = get_non_connecting_pipeline_executor(dummy_pipeline_options)
    for batch in batches:
        run_batch_triggers_and_validate(pe, batch)


# def test__handle_triggers_within_batch_trigger_timespan_across_batch() -> None:
#     test_inform_selector_and_trigger.side_effect = [(0, {}), (1, {}), (2, {})]

#     # each tuple is (key, timestamp, label)
#     first_batch = [
#         # trigger 0
#         (10, 1, 5),
#         (11, 2, 5),
#     ]
#     triggering_indices = []
#     pe._handle_triggers_within_batch(first_batch, triggering_indices)
#     assert pe.remaining_data_range == (1, 2)
#     test_get_number_of_samples.assert_not_called()
#     test_inform_selector_and_trigger.assert_not_called()
#     test__run_training.assert_not_called()
#     test_inform_selector.assert_called_once_with(PIPELINE_ID, first_batch)

#     reset_state()
#     second_batch = [
#         # trigger 1
#         (12, 3, 5),
#         (13, 4, 5),
#     ]
#     triggering_indices = [-1]
#     test_get_number_of_samples.side_effect = [2]  # the size of trigger 0
#     pe._handle_triggers_within_batch(second_batch, triggering_indices)
#     assert pe.pipeline_log["supervisor"]["triggers"][0]["first_timestamp"] == 1
#     assert pe.pipeline_log["supervisor"]["triggers"][0]["last_timestamp"] == 2
#     assert pe.remaining_data_range == (3, 4)
#     test_get_number_of_samples.assert_called_once_with(PIPELINE_ID, 0)
#     test_inform_selector_and_trigger.assert_called_once_with(42, [])
#     test__run_training.assert_called_once_with(0)
#     test_inform_selector.assert_called_once_with(PIPELINE_ID, second_batch)

#     reset_state()
#     third_batch = [
#         # still trigger 1's data
#         (14, 5, 5),
#         (15, 6, 5),
#         (16, 7, 5),
#     ]
#     triggering_indices = []
#     pe._handle_triggers_within_batch(third_batch, triggering_indices)
#     assert pe.remaining_data_range == (3, 7)
#     reset_state()

#     fourth_batch = [
#         # still trigger 1's data
#         (17, 8, 5),
#         (18, 9, 5),
#         # trigger 2
#         (19, 10, 5),
#         (20, 11, 5),
#     ]

#     triggering_indices = [1, 3]
#     test_get_number_of_samples.side_effect = [7, 2]
#     pe._handle_triggers_within_batch(fourth_batch, triggering_indices)
#     assert pe.remaining_data_range is None
#     assert pe.pipeline_log["supervisor"]["triggers"][1]["first_timestamp"] == 3
#     assert pe.pipeline_log["supervisor"]["triggers"][1]["last_timestamp"] == 9
#     assert pe.pipeline_log["supervisor"]["triggers"][2]["first_timestamp"] == 10
#     assert pe.pipeline_log["supervisor"]["triggers"][2]["last_timestamp"] == 11


@pytest.mark.parametrize("evaluate", [True, False])
@pytest.mark.parametrize(
    "trigger_id, training_id, model_id",
    [
        (21, 1337, 101),
    ],
)
@patch.object(GRPCHandler, "store_evaluation_results")
@patch.object(GRPCHandler, "wait_for_evaluation_completion")
@patch.object(GRPCHandler, "start_evaluation")
@patch.object(GRPCHandler, "store_trained_model")
@patch.object(GRPCHandler, "wait_for_training_completion")
@patch.object(GRPCHandler, "start_training")
def test_train_and_evaluate(
    test_start_training: MagicMock,
    test_wait_for_training_completion: MagicMock,
    test_store_trained_model: MagicMock,
    test_start_evaluation: MagicMock,
    test_wait_for_evaluation_completion: MagicMock,
    test_store_evaluation_results: MagicMock,
    evaluate: bool,
    trigger_id: int,
    training_id: int,
    model_id: int,
    dummy_pipeline_options: PipelineOptions,
    pipeline_evaluation_config: EvaluationConfig,
):
    dummy_pipeline_options.pipeline_id = 42
    if evaluate:
        pipeline_evaluation_config.result_writers = ["json"]
        dummy_pipeline_options.pipeline_config.evaluation = pipeline_evaluation_config
        dummy_pipeline_options.eval_directory = EVALUATION_DIRECTORY
    else:
        dummy_pipeline_options.pipeline_config.evaluation = None

    pe = get_non_connecting_pipeline_executor(dummy_pipeline_options)

    evaluations = {1: EvaluationStatusReporter(TRAINING_STATUS_QUEUE, EVAL_ID, "MNIST_eval", 1000)}

    test_start_training.return_value = training_id
    test_store_trained_model.return_value = model_id
    test_wait_for_training_completion.return_value = {}

    if evaluate:
        test_start_evaluation.return_value = evaluations

    ret_training_id, ret_model_id = pe._train_and_store_model(pe.state, pe.logs, trigger_id)
    if evaluate:
        pe._evaluate_and_store_results(pe.state, pe.logs, trigger_id, training_id, model_id)

    assert (ret_training_id, ret_model_id) == (training_id, model_id)
    assert pe.state.previous_model_id == model_id
    assert pe.state.current_training_id == training_id

    test_wait_for_training_completion.assert_called_once_with(training_id, 42, trigger_id)
    test_start_training.assert_called_once_with(42, trigger_id, pe.state.pipeline_config, None, None)
    test_wait_for_training_completion.assert_called_once_with(training_id, 42, trigger_id)
    test_store_trained_model.assert_called_once_with(training_id)

    if evaluate:
        test_start_evaluation.assert_called_once_with(model_id, pe.state.pipeline_config)
        test_wait_for_evaluation_completion.assert_called_once_with(training_id, evaluations)
        test_store_evaluation_results.assert_called_once_with(ANY, evaluations)
        assert len(test_store_evaluation_results.call_args[0][0]) == 1
        result_writer: AbstractEvaluationResultWriter = test_store_evaluation_results.call_args[0][0][0]
        assert result_writer.eval_directory == EVALUATION_DIRECTORY
        assert result_writer.pipeline_id == 42
        assert result_writer.trigger_id == 21

    else:
        test_start_evaluation.assert_not_called()


@patch.object(GRPCHandler, "store_trained_model", return_value=101)
@patch.object(GRPCHandler, "start_training", return_value=1337)
@patch.object(GRPCHandler, "wait_for_training_completion", return_value={})
def test_run_training_set_num_samples_to_pass(
    test_wait_for_training_completion: MagicMock,
    test_start_training: MagicMock,
    test_store_trained_model: MagicMock,
    dummy_pipeline_options: PipelineOptions,
):
    dummy_pipeline_options.pipeline_id = 42
    dummy_pipeline_options.pipeline_config.training.num_samples_to_pass = [73]
    pe = get_non_connecting_pipeline_executor(dummy_pipeline_options)

    pe._train_and_store_model(pe.state, pe.logs, trigger_id=21)
    test_start_training.assert_called_once_with(42, 21, ANY, None, 73)
    test_start_training.reset_mock()

    # trigger is added to trigger list _execute_triggers, as we are not calling it here, we fake the it
    pe.state.triggers.append(21)

    # the next time _run_training is called, the num_samples_to_pass should be set to 0
    # because the next trigger is out of the range of `num_samples_to_pass`
    pe._train_and_store_model(pe.state, pe.logs, trigger_id=22)
    test_start_training.assert_called_once_with(42, 22, ANY, 101, None)


@pytest.mark.parametrize("stop_replay_at", [None, 2, 43])
@pytest.mark.parametrize(
    "grpc_fetch_retval",
    [
        [([(10, 1, 0), (11, 2, 0)], 0)],  # open_interval
        [([(10, 1, 0)], 0), ([(11, 2, 0)], 0)],  # open_interval_batched
        [([(10, 1, 0)], 0), ([(11, 2, 0)], 0), ([], 0)],
    ],
)
@patch.object(PipelineExecutor, "_process_new_data")
@patch.object(GRPCHandler, "get_new_data_since")
@patch.object(GRPCHandler, "get_data_in_interval")
def test_replay_data(
    test_get_data_in_interval: MagicMock,
    test_get_new_data_since: MagicMock,
    test__process_new_data: MagicMock,
    grpc_fetch_retval: list[tuple[list[tuple[int, int]], int]],
    stop_replay_at: int | None,
    dummy_pipeline_options: PipelineOptions,
):
    dummy_pipeline_options.pipeline_id = 42
    dummy_pipeline_options.start_replay_at = 0
    dummy_pipeline_options.stop_replay_at = stop_replay_at
    pe = get_non_connecting_pipeline_executor(dummy_pipeline_options)

    if stop_replay_at:
        test_get_data_in_interval.side_effect = [grpc_fetch_retval]
    else:
        test_get_new_data_since.side_effect = [grpc_fetch_retval]

    pe._replay_data(pe.state, pe.logs)

    if stop_replay_at:
        test_get_data_in_interval.assert_called_once_with("test", 0, stop_replay_at)
    else:
        test_get_new_data_since.assert_called_once_with("test", 0)

    assert test__process_new_data.call_count == len(grpc_fetch_retval)
    assert test__process_new_data.call_args_list == [call(ANY, ANY, r[0], r[1]) for r in grpc_fetch_retval]


@pytest.mark.parametrize("experiment", [True, False])
@patch.object(PipelineExecutor, "_replay_data", return_value=None)
@patch.object(PipelineExecutor, "_serve_online_data", return_value=None)
@patch.object(GRPCHandler, "init_cluster_connection", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
def test_execute_pipeline(
    test_grpc_connection_established: MagicMock,
    test_init_cluster_connection: MagicMock,
    test_serve_online_data: MagicMock,
    test_replay_data: MagicMock,
    experiment: bool,
    dummy_pipeline_options: PipelineOptions,
):
    dummy_pipeline_options.start_replay_at = 0 if experiment else None

    execute_pipeline(dummy_pipeline_options)

    test_init_cluster_connection.assert_called_once()

    if experiment:
        test_serve_online_data.assert_not_called()
        test_replay_data.assert_called_once()
    else:
        test_serve_online_data.assert_called_once()
        test_replay_data.assert_not_called()
