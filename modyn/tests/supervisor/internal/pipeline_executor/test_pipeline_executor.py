# pylint: disable=unused-argument,redefined-outer-name
import datetime
import multiprocessing as mp
import os
import pathlib
import shutil
import time
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import overload
from unittest import mock
from unittest.mock import ANY, MagicMock, PropertyMock, call, patch

import pytest

from modyn.config.schema.pipeline import EvaluationConfig, ModynPipelineConfig
from modyn.config.schema.system import DatasetsConfig, ModynConfig, SupervisorConfig

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    EvaluateModelIntervalResponse,
    EvaluateModelResponse,
    EvaluationAbortedReason,
    EvaluationIntervalData,
    SingleMetricResult,
)
from modyn.supervisor.internal.eval.strategies.abstract import EvalInterval
from modyn.supervisor.internal.eval.strategies.slicing import SlicingEvalStrategy
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.pipeline_executor import (
    PipelineExecutor,
    execute_pipeline,
)
from modyn.supervisor.internal.pipeline_executor.models import (
    ConfigLogs,
    ExecutionState,
    MultiEvaluationInfo,
    PipelineExecutionParams,
    PipelineLogs,
    StageInfo,
    StageLog,
)
from modyn.supervisor.internal.pipeline_executor.pipeline_executor import (
    _pipeline_stage_parents,
    pipeline_stage,
)

EVALUATION_DIRECTORY: pathlib.Path = pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir"
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
def minimal_pipeline_config(
    dummy_pipeline_config: ModynPipelineConfig,
) -> ModynPipelineConfig:
    config = dummy_pipeline_config.model_copy()
    return config


def get_dummy_pipeline_args(
    system_config: ModynConfig, pipeline_config: ModynPipelineConfig
) -> PipelineExecutionParams:
    return PipelineExecutionParams(
        start_timestamp=START_TIMESTAMP,
        pipeline_id=PIPELINE_ID,
        modyn_config=system_config,
        pipeline_config=pipeline_config,
        eval_directory=EVALUATION_DIRECTORY,
        exception_queue=EXCEPTION_QUEUE,
        pipeline_status_queue=PIPELINE_STATUS_QUEUE,
        training_status_queue=TRAINING_STATUS_QUEUE,
        eval_status_queue=EVAL_STATUS_QUEUE,
    )


@pytest.fixture
def dummy_pipeline_args(
    minimal_system_config: ModynConfig, minimal_pipeline_config: ModynPipelineConfig
) -> PipelineExecutionParams:
    return get_dummy_pipeline_args(minimal_system_config, minimal_pipeline_config)


@pytest.fixture
def dummy_execution_state(
    dummy_pipeline_args: PipelineExecutionParams,
) -> ExecutionState:
    return ExecutionState(**vars(dummy_pipeline_args))


@pytest.fixture
def dummy_logs(dummy_pipeline_args: PipelineExecutionParams) -> PipelineLogs:
    options = dummy_pipeline_args
    return PipelineLogs(
        pipeline_id=PIPELINE_ID,
        pipeline_stages=_pipeline_stage_parents,
        config=ConfigLogs(system=options.modyn_config, pipeline=options.pipeline_config),
        experiment=options.experiment_mode,
        start_replay_at=options.start_replay_at,
        stop_replay_at=options.stop_replay_at,
    )


@pytest.fixture
def dummy_stage_log() -> StageLog:
    return StageLog(
        id="dummy",
        id_seq_num=-1,
        start=0,
        sample_idx=1,
        sample_time=1000,
        trigger_idx=0,
    )


@overload
def get_non_connecting_pipeline_executor(
    system_config: ModynConfig, pipeline_config: ModynPipelineConfig
) -> PipelineExecutor: ...


@overload
def get_non_connecting_pipeline_executor(
    pipeline_args: PipelineExecutionParams,
) -> PipelineExecutor: ...


def get_non_connecting_pipeline_executor(
    pipeline_args: PipelineExecutionParams | None = None,
    system_config: ModynConfig | None = None,
    pipeline_config: ModynPipelineConfig | None = None,
) -> PipelineExecutor:
    if pipeline_args:
        return PipelineExecutor(pipeline_args)
    return PipelineExecutor(get_dummy_pipeline_args(system_config, pipeline_config))


@pytest.fixture
def non_connecting_pipeline_executor(
    dummy_pipeline_args: PipelineExecutionParams,
) -> PipelineExecutor:
    return PipelineExecutor(dummy_pipeline_args)


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


def test_pipeline_stage_decorator(dummy_pipeline_args: PipelineExecutionParams) -> None:
    class TestStageLogInfo(StageInfo):
        name: str

    class TestPipelineExecutor(PipelineExecutor):
        @pipeline_stage(PipelineStage.INIT, log=True, track=True)
        def _stage_func(self, s: ExecutionState, log: StageLog) -> int:
            time.sleep(0.1)
            log.info = TestStageLogInfo(name="test")

            return 1

    pe = TestPipelineExecutor(dummy_pipeline_args)
    t0 = datetime.datetime.now()
    assert pe._stage_func(pe.state, pe.logs) == 1
    t1 = datetime.datetime.now()

    assert len(pe.logs.supervisor_logs.stage_runs) == 1
    assert pe.logs.supervisor_logs.stage_runs[0].info.name == "test"
    assert (pe.logs.supervisor_logs.stage_runs[0].start - t0).total_seconds() < 8e-3
    assert (pe.logs.supervisor_logs.stage_runs[0].end - t1).total_seconds() < 8e-3
    assert abs((pe.logs.supervisor_logs.stage_runs[0].duration - (t1 - t0)).total_seconds()) < 8e-3


def test_pipeline_stage_decorator_generator(
    dummy_pipeline_args: PipelineExecutionParams,
) -> None:
    class TestStageLogInfo(StageInfo):
        elements: list[int]
        finalized: bool = False

    def create_generator(x: int = 3) -> Generator[int, None, None]:
        time.sleep(0.1)
        for i in range(x):
            time.sleep(0.02)
            yield i

    class TestPipelineExecutor(PipelineExecutor):
        @pipeline_stage(PipelineStage.INIT, log=True, track=True)
        def _stage_func(self, s: ExecutionState, log: StageLog) -> Generator[int, None, None]:
            try:
                time.sleep(0.1)
                log.info = TestStageLogInfo(elements=[])
                for i in create_generator():
                    time.sleep(0.02)
                    log.info.elements.append(i)
                    yield i
            finally:
                log.info.finalized = True

    times: list[datetime.timedelta] = []
    gen_values: list[int] = []

    last_time = datetime.datetime.now()
    pe = TestPipelineExecutor(dummy_pipeline_args)
    gen = pe._stage_func(pe.state, pe.logs)
    times.append(datetime.datetime.now() - last_time)
    last_time = datetime.datetime.now()
    for x in gen:
        gen_values.append(x)
        times.append(datetime.datetime.now() - last_time)
        time.sleep(0.2)  # this should not be counted in the time of the generator
        last_time = datetime.datetime.now()

        if x == 2:
            break  # stop the generator early

    gen = None  # make sure the generator is closed

    assert gen_values == [0, 1, 2]
    assert len(times) == 4

    assert pe.logs.supervisor_logs.stage_runs[0].info.elements == [0, 1, 2]
    assert pe.logs.supervisor_logs.stage_runs[0].info.finalized

    assert abs(times[0].total_seconds()) < 0.005
    assert abs(times[1].total_seconds() - 0.1 * 2 - 0.02 * 2) < 0.035  # higher tolerance due to pipeline_stage overhead
    assert abs(times[2].total_seconds() - 0.02 * 2) < 0.025
    assert abs(times[3].total_seconds() - 0.02 * 2) < 0.025


def test_get_dataset_selector_batch_size_given(
    minimal_system_config: ModynConfig,
    minimal_pipeline_config: ModynPipelineConfig,
    dummy_dataset_config: DatasetsConfig,
) -> None:
    dataset1 = dummy_dataset_config.model_copy()
    dataset1.selector_batch_size = 2048
    minimal_system_config.storage.datasets = [dataset1]
    pe = get_non_connecting_pipeline_executor(
        system_config=minimal_system_config, pipeline_config=minimal_pipeline_config
    )
    assert pe.state.selector_batch_size == 2048


def test_shutdown_trainer() -> None:
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


@patch.object(
    GRPCHandler,
    "get_new_data_since",
    return_value=[([(10, 42, 0)], 98), ([(11, 43, 1)], 99)],
)
@patch.object(PipelineExecutor, "_process_new_data", side_effect=[[10], [11]])
def test_fetch_new_data_batched(
    test__process_new_data: MagicMock,
    test_get_new_data_since: MagicMock,
    non_connecting_pipeline_executor: PipelineExecutor,
    dummy_execution_state: ExecutionState,
    dummy_logs: PipelineLogs,
) -> None:
    dummy_execution_state.max_timestamp = 21
    triggers = non_connecting_pipeline_executor._fetch_new_data(dummy_execution_state, dummy_logs)
    test_get_new_data_since.assert_called_once_with("test", 21)

    expected_calls = [
        call(ANY, ANY, [(10, 42, 0)], 98),
        call(ANY, ANY, [(11, 43, 1)], 99),
    ]
    assert test__process_new_data.call_args_list == expected_calls
    assert triggers == 2


def test_serve_online_data(
    non_connecting_pipeline_executor: PipelineExecutor,
    dummy_execution_state: ExecutionState,
    dummy_logs: PipelineLogs,
) -> None:
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

            expected_get_new_data_arg_list = [
                call("test", 21),
                call("test", 43),
                call("test", 45),
                call("test", 45),
            ]
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
) -> None:
    pe = non_connecting_pipeline_executor
    new_data = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (15, 6), (16, 7), (17, 8)]

    batch_mock: MagicMock
    with patch.object(ExecutionState, "selector_batch_size", new_callable=PropertyMock) as selector_batch_size_mock:
        selector_batch_size_mock.return_value = selector_batch_size
        with patch.object(
            PipelineExecutor,
            "_process_new_data_batch",
            side_effect=batching_return_vals,
        ) as batch_mock:
            trigger_indexes = pe._process_new_data(dummy_execution_state, pe.logs, new_data, 0)
            assert trigger_indexes == [x for sub in batching_return_vals for x in sub]
            assert batch_mock.call_args_list == (expected_process_new_data_batch_args or [call(ANY, ANY, new_data)])


@patch.object(GRPCHandler, "inform_selector", return_value={})
def test__process_new_data_batch_no_triggers(
    test_inform_selector: MagicMock, dummy_pipeline_args: PipelineExecutionParams
) -> None:
    dummy_pipeline_args.pipeline_id = 42
    pe = get_non_connecting_pipeline_executor(dummy_pipeline_args)
    batch = [(10, 1), (11, 2)]

    with patch.object(pe.trigger, "inform") as inform_mock:

        def fake(self, *args, **kwargs):
            yield from []

        inform_mock.side_effect = fake
        assert len(pe._process_new_data_batch(pe.state, pe.logs, batch)) == 0

        inform_mock.assert_called_once()
        inform_mock.call_args_list == [call(batch, ANY)]
        test_inform_selector.assert_called_once_with(42, batch)


@pytest.mark.parametrize(
    (
        "batch,trigger_indexes,trigger_ids,inform_selector_and_trigger_expected_args,"
        "train_and_store_model_expected_args,inform_selector_expected_args"
    ),
    [
        (
            [
                (10, 1, 0),
                (11, 2, 0),
                (12, 3, 0),
                (13, 4, 0),
                (14, 5, 0),
                (15, 6, 0),
                (16, 7, 0),
            ],  # batch
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
            [
                (10, 1, 5),
                (11, 2, 5),
                (12, 3, 5),
                (13, 4, 5),
                (14, 5, 5),
                (15, 6, 5),
                (16, 7, 5),
            ],  # batch
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
    trigger_ids: list[tuple[int, dict]],
    inform_selector_and_trigger_expected_args: list,
    train_and_store_model_expected_args: list,
    inform_selector_expected_args: list,
    dummy_pipeline_args: PipelineExecutionParams,
) -> None:
    dummy_pipeline_args.pipeline_id = 42
    dummy_pipeline_args.pipeline_config.evaluation = None
    pe = get_non_connecting_pipeline_executor(dummy_pipeline_args)

    test_inform_selector_and_trigger.side_effect = trigger_ids
    test_inform_selector.return_value = {}
    test_get_number_of_samples.side_effect = [len(params) for params in inform_selector_and_trigger_expected_args]

    pe._handle_triggers(pe.state, pe.logs, batch, trigger_indexes)
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
    dummy_pipeline_args: PipelineExecutionParams,
) -> None:
    tracking_columns = [
        "trigger_i",
        "trigger_id",
        "trigger_index",
        "first_timestamp",
        "last_timestamp",
    ]
    test_inform_selector_and_trigger.side_effect = inform_selector_and_trigger_retval
    test_inform_selector.return_value = {}

    def run_batch_triggers_and_validate(pe: PipelineExecutor, batch: _BatchConfig) -> None:
        test_get_number_of_samples.reset_mock()
        test_inform_selector_and_trigger.reset_mock()
        test__run_training.reset_mock()
        test_inform_selector.reset_mock()

        test_get_number_of_samples.side_effect = batch.get_num_samples

        pe._handle_triggers(pe.state, pe.logs, batch.data, batch.trigger_indexes)
        pe._inform_selector_remaining_data(pe.state, pe.logs, batch.data, batch.trigger_indexes)

        assert pe.state.remaining_data_range == batch.remaining_data_range
        if batch.expected_tracking:
            assert pe.state.tracking[PipelineStage.HANDLE_SINGLE_TRIGGER.name][tracking_columns].to_dict("list") == (
                batch.expected_tracking
            )

    pe = get_non_connecting_pipeline_executor(dummy_pipeline_args)
    for batch in batches:
        run_batch_triggers_and_validate(pe, batch)


@pytest.mark.parametrize(
    "trigger_id, training_id, model_id",
    [
        (21, 1337, 101),
    ],
)
@patch.object(GRPCHandler, "store_trained_model")
@patch.object(GRPCHandler, "wait_for_training_completion")
@patch.object(GRPCHandler, "start_training")
@patch.object(GRPCHandler, "get_number_of_samples", return_value=34)
@patch.object(GRPCHandler, "get_status_bar_scale", return_value=40)
def test_train_and_store_model(
    test_get_status_bar_scale: MagicMock,
    test_get_number_of_samples: MagicMock,
    test_start_training: MagicMock,
    test_wait_for_training_completion: MagicMock,
    test_store_trained_model: MagicMock,
    trigger_id: int,
    training_id: int,
    model_id: int,
    dummy_pipeline_args: PipelineExecutionParams,
) -> None:
    dummy_pipeline_args.pipeline_id = 42
    dummy_pipeline_args.pipeline_config.evaluation = None
    pe = get_non_connecting_pipeline_executor(dummy_pipeline_args)

    test_start_training.return_value = training_id
    test_store_trained_model.return_value = model_id
    test_wait_for_training_completion.return_value = {
        "num_batches": 0,
        "num_samples": 0,
    }

    ret_training_id, ret_model_id = pe._train_and_store_model(pe.state, pe.logs, trigger_id)
    assert (ret_training_id, ret_model_id) == (training_id, model_id)
    assert pe.state.previous_model_id == model_id
    assert pe.state.current_training_id == training_id

    test_wait_for_training_completion.assert_called_once_with(training_id, ANY)
    test_start_training.assert_called_once_with(42, trigger_id, ANY, ANY, None, None)
    test_store_trained_model.assert_called_once_with(training_id)


@patch.object(GRPCHandler, "store_trained_model", return_value=101)
@patch.object(GRPCHandler, "start_training", return_value=1337)
@patch.object(
    GRPCHandler,
    "wait_for_training_completion",
    return_value={"num_batches": 0, "num_samples": 0},
)
@patch.object(GRPCHandler, "get_number_of_samples", return_value=34)
@patch.object(GRPCHandler, "get_status_bar_scale", return_value=40)
def test_run_training_set_num_samples_to_pass(
    test_get_status_bar_scale: MagicMock,
    test_get_number_of_samples: MagicMock,
    test_wait_for_training_completion: MagicMock,
    test_start_training: MagicMock,
    test_store_trained_model: MagicMock,
    dummy_pipeline_args: PipelineExecutionParams,
) -> None:
    dummy_pipeline_args.pipeline_id = 42
    dummy_pipeline_args.pipeline_config.training.num_samples_to_pass = [73]
    pe = get_non_connecting_pipeline_executor(dummy_pipeline_args)

    pe._train_and_store_model(pe.state, pe.logs, trigger_id=21)
    test_start_training.assert_called_once_with(42, 21, ANY, ANY, None, 73)
    test_start_training.reset_mock()

    # trigger is added to trigger list _execute_triggers, as we are not calling it here, we fake the id
    pe.state.triggers.append(21)

    # the next time _run_training is called, the num_samples_to_pass should be set to 0
    # because the next trigger is out of the range of `num_samples_to_pass`
    pe._train_and_store_model(pe.state, pe.logs, trigger_id=22)
    test_start_training.assert_called_once_with(42, 22, ANY, ANY, 101, None)


@pytest.mark.parametrize("test_failure", [False, True])
@patch.object(
    GRPCHandler,
    "wait_for_evaluation_completion",
    return_value={"num_batches": 0, "num_samples": 0},
)
@patch.object(GRPCHandler, "cleanup_evaluations")
@patch.object(GRPCHandler, "get_evaluation_results")
def test__start_evaluations(
    test_get_evaluation_results: MagicMock,
    test_cleanup_evaluations: MagicMock,
    test_wait_for_evaluation_completion: MagicMock,
    test_failure: bool,
    dummy_pipeline_args: PipelineExecutionParams,
    pipeline_evaluation_config: EvaluationConfig,
) -> None:
    eval_dataset_config = pipeline_evaluation_config.datasets[0]
    dummy_pipeline_args.pipeline_config.evaluation = pipeline_evaluation_config

    evaluator_stub_mock = mock.Mock(spec=["evaluate_model"])

    success_interval = EvaluateModelIntervalResponse(
        eval_aborted_reason=EvaluationAbortedReason.NOT_ABORTED, dataset_size=10
    )
    failure_interval = EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.EMPTY_DATASET)

    pe = get_non_connecting_pipeline_executor(dummy_pipeline_args)

    if test_failure:

        def get_eval_intervals(
            training_intervals: list[tuple[int, int]],
            dataset_end_time: int | None = None,
        ) -> Iterable[EvalInterval]:
            yield from [
                EvalInterval(start=0, end=100, active_model_trained_before=50),
                EvalInterval(start=100, end=200, active_model_trained_before=150),
                EvalInterval(start=200, end=300, active_model_trained_before=250),
            ]

        evaluator_stub_mock.evaluate_model.side_effect = [
            EvaluateModelResponse(
                evaluation_started=True,
                evaluation_id=42,
                interval_responses=[
                    success_interval,
                    failure_interval,
                    success_interval,
                ],
            )
        ]
        test_get_evaluation_results.return_value = [
            EvaluationIntervalData(
                interval_index=idx, evaluation_data=[SingleMetricResult(metric="Accuracy", result=0.5)]
            )
            for idx in [0, 2]
        ]

    else:
        intervals = [
            (0, 100),
            (100, 200),
            (0, None),
            (0, 200),
            (0, 0),
            (200, None),
            (0, None),
            (0, 0),
        ]

        def get_eval_intervals(
            training_intervals: list[tuple[int, int]],
            dataset_end_time: int | None = None,
        ) -> Iterable[EvalInterval]:
            yield from [EvalInterval(start=start, end=end, active_model_trained_before=0) for start, end in intervals]

        evaluator_stub_mock.evaluate_model.return_value = EvaluateModelResponse(
            evaluation_started=True,
            evaluation_id=42,
            interval_responses=[success_interval for _ in range(len(intervals))],
        )
        test_get_evaluation_results.return_value = [
            EvaluationIntervalData(
                interval_index=idx, evaluation_data=[SingleMetricResult(metric="Accuracy", result=0.5)]
            )
            for idx in range(len(intervals))
        ]

    pe.grpc.evaluator = evaluator_stub_mock

    with patch.object(SlicingEvalStrategy, "get_eval_intervals", side_effect=get_eval_intervals):
        model_id = 1
        pe._evaluate_and_store_results(
            pe.state,
            pe.logs,
            trigger_id=0,
            training_id=0,
            model_id=model_id,
            first_timestamp=20,
            last_timestamp=70,
        )

        assert evaluator_stub_mock.evaluate_model.call_count == 1  # batched
        if test_failure:
            assert test_cleanup_evaluations.call_count == 2
            assert test_wait_for_evaluation_completion.call_count == 1

            stage_info = [
                run.info for run in pe.logs.supervisor_logs.stage_runs if isinstance(run.info, MultiEvaluationInfo)
            ]
            assert len(stage_info) == 1
            assert len(stage_info[0].interval_results) == 3
            assert (
                stage_info[0].interval_results[0].eval_request.interval_start,
                stage_info[0].interval_results[0].eval_request.interval_end,
            ) == (0, 100)
            assert (
                stage_info[0].interval_results[1].eval_request.interval_start,
                stage_info[0].interval_results[1].eval_request.interval_end,
            ) == (100, 200)
            assert (
                stage_info[0].interval_results[2].eval_request.interval_start,
                stage_info[0].interval_results[2].eval_request.interval_end,
            ) == (200, 300)
            assert stage_info[0].interval_results[0].failure_reason is None
            assert stage_info[0].interval_results[1].failure_reason == "EMPTY_DATASET"
            assert stage_info[0].interval_results[2].failure_reason is None
        else:
            expected_calls = [
                call(
                    GRPCHandler.prepare_evaluation_request(
                        eval_dataset_config.model_dump(by_alias=True),
                        model_id,
                        pipeline_evaluation_config.device,
                        [(start_ts, end_ts) for start_ts, end_ts in intervals],
                    )
                )
            ]

            # because of threadpool, ordering isn't guaranteed
            assert len(evaluator_stub_mock.evaluate_model.call_args_list) == len(expected_calls)
            assert all([eval_call in expected_calls for eval_call in evaluator_stub_mock.evaluate_model.call_args_list])


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
    dummy_pipeline_args: PipelineExecutionParams,
) -> None:
    dummy_pipeline_args.pipeline_id = 42
    dummy_pipeline_args.start_replay_at = 0
    dummy_pipeline_args.stop_replay_at = stop_replay_at
    pe = get_non_connecting_pipeline_executor(dummy_pipeline_args)

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
    dummy_pipeline_args: PipelineExecutionParams,
) -> None:
    dummy_pipeline_args.start_replay_at = 0 if experiment else None

    execute_pipeline(dummy_pipeline_args)

    test_init_cluster_connection.assert_called_once()

    if experiment:
        test_serve_online_data.assert_not_called()
        test_replay_data.assert_called_once()
    else:
        test_serve_online_data.assert_called_once()
        test_replay_data.assert_not_called()
