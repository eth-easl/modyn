import datetime
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from modyn.config.schema.pipeline.config import ModynPipelineConfig
from modyn.config.schema.pipeline.evaluation.config import EvaluationConfig
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.slicing import SlicingEvalStrategyConfig
from modyn.config.schema.system.config import ModynConfig
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    EvaluateModelIntervalResponse,
    EvaluateModelResponse,
    EvaluationAbortedReason,
    EvaluationIntervalData,
    SingleMetricResult,
)
from modyn.supervisor.internal.eval.handler import EvalHandler, EvalRequest
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.pipeline_executor.evaluation_executor import EvalStateConfig, EvaluationExecutor
from modyn.supervisor.internal.pipeline_executor.models import StageLog, SupervisorLogs


@pytest.fixture
def pipeline_config(
    dummy_pipeline_config: ModynPipelineConfig, pipeline_evaluation_config: EvaluationConfig
) -> ModynPipelineConfig:
    dummy_pipeline_config.evaluation = pipeline_evaluation_config
    return dummy_pipeline_config


@pytest.fixture(scope="module")
def tmp_dir_tests() -> Iterator[Path]:
    tmpdir = TemporaryDirectory()
    Path(tmpdir.name).mkdir(exist_ok=True)
    yield Path(tmpdir.name)
    tmpdir.cleanup()


@pytest.fixture
def eval_state_config(
    dummy_system_config: ModynConfig, pipeline_config: ModynPipelineConfig, tmp_dir_tests: Path
) -> EvalStateConfig:
    return EvalStateConfig(
        pipeline_id=1,
        eval_dir=tmp_dir_tests,
        config=dummy_system_config,
        pipeline=pipeline_config,
    )


@pytest.fixture
def evaluation_executor(
    dummy_system_config: ModynConfig,
    pipeline_config: ModynPipelineConfig,
    eval_state_config: EvalStateConfig,
    tmp_dir_tests: Path,
) -> EvaluationExecutor:
    return EvaluationExecutor(
        pipeline_id=1,
        pipeline_logdir=tmp_dir_tests,
        config=dummy_system_config,
        pipeline=pipeline_config,
        grpc_handler=GRPCHandler(eval_state_config.config.model_dump(by_alias=True)),
    )


@pytest.fixture
def tracking_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pipeline_id": [1],
            "trigger_id": [1],
            "first_timestamp": [1],
            "last_timestamp": [2],
            "id_model": [1],
            "training_id": [1],
        }
    )


@patch.object(GRPCHandler, "init_cluster_connection", return_value=None)
def test_evaluation_executor_state_management(
    test_init_cluster_connection: MagicMock,
    evaluation_executor: EvaluationExecutor,
    tracking_df: pd.DataFrame,
    tmp_dir_tests: Path,
) -> None:
    evaluation_executor.register_tracking_info(
        {
            PipelineStage.HANDLE_SINGLE_TRIGGER.name: tracking_df,
            PipelineStage.STORE_TRAINED_MODEL.name: tracking_df,
        },
        dataset_end_time=99,
    )
    evaluation_executor.create_snapshot()

    # assert snapshot
    assert (tmp_dir_tests / "snapshot").exists()
    assert (tmp_dir_tests / "snapshot" / "eval_state.yaml").exists()
    assert (tmp_dir_tests / "snapshot" / "context.pcl").exists()

    test_init_cluster_connection.assert_not_called()
    loaded_eval_executor = EvaluationExecutor.init_from_path(tmp_dir_tests)
    test_init_cluster_connection.assert_called_once()

    assert loaded_eval_executor.pipeline_id == evaluation_executor.pipeline_id
    assert loaded_eval_executor.pipeline_logdir == evaluation_executor.pipeline_logdir
    assert loaded_eval_executor.config == evaluation_executor.config
    assert loaded_eval_executor.pipeline == evaluation_executor.pipeline
    assert loaded_eval_executor.grpc is not None

    assert loaded_eval_executor.context.tracking_dfs.keys() == evaluation_executor.context.tracking_dfs.keys()
    for key in evaluation_executor.context.tracking_dfs.keys():
        assert loaded_eval_executor.context.tracking_dfs[key].equals(evaluation_executor.context.tracking_dfs[key])

    assert len(loaded_eval_executor.eval_handlers) == len(evaluation_executor.eval_handlers)


def dummy_eval_request() -> EvalRequest:
    return EvalRequest(
        trigger_id=1,
        training_id=1,
        id_model=1,
        currently_active_model=True,
        currently_trained_model=False,
        dataset_id="MNIST_eval",
        eval_handler="e",
    )


@pytest.fixture
def dummy_stage_log() -> StageLog:
    return StageLog(
        id="log",
        id_seq_num=-1,
        start=datetime.datetime(2021, 1, 1),
        batch_idx=-1,
        sample_idx=-1,
        sample_time=-1,
        trigger_idx=-1,
    )


@patch.object(EvalHandler, "get_eval_requests_after_pipeline", return_value=[dummy_eval_request()])
@patch.object(EvalHandler, "get_eval_requests_after_training", return_value=[dummy_eval_request()])
@patch.object(EvaluationExecutor, "_launch_evaluations_async", return_value=[SupervisorLogs(stage_runs=[])])
def test_evaluation_handler_post_training(
    test_launch_evaluations_async: Any,
    test_get_eval_requests_after_training: Any,
    test_get_eval_requests_after_pipeline: Any,
    evaluation_executor: EvaluationExecutor,
    tracking_df: pd.DataFrame,
    dummy_stage_log: StageLog,
) -> None:
    evaluation_executor.eval_handlers = [
        EvalHandler(
            EvalHandlerConfig(
                execution_time="after_training",
                strategy=SlicingEvalStrategyConfig(eval_every="100s", eval_start_from=0, eval_end_at=300),
                models="matrix",
                datasets=["dataset1"],
            )
        ),
        EvalHandler(
            EvalHandlerConfig(
                execution_time="after_pipeline",
                strategy=SlicingEvalStrategyConfig(eval_every="100s", eval_start_from=0, eval_end_at=300),
                models="matrix",
                datasets=["dataset2"],
            )
        ),
    ]

    evaluation_executor.grpc.evaluator = 1  # type: ignore
    evaluation_executor.run_pipeline_evaluations(
        log=dummy_stage_log,
        trigger_id=-1,
        training_id=-1,
        model_id=-1,
        first_timestamp=1,
        last_timestamp=3,
    )

    test_get_eval_requests_after_training.assert_called_once()
    test_get_eval_requests_after_pipeline.assert_not_called()
    test_launch_evaluations_async.assert_called_once()

    # reset mocks
    test_get_eval_requests_after_training.reset_mock()
    test_get_eval_requests_after_pipeline.reset_mock()
    test_launch_evaluations_async.reset_mock()

    evaluation_executor.register_tracking_info(
        {
            PipelineStage.HANDLE_SINGLE_TRIGGER.name: tracking_df,
            PipelineStage.STORE_TRAINED_MODEL.name: tracking_df,
        },
        dataset_end_time=99,
    )
    evaluation_executor.run_post_pipeline_evaluations()
    test_get_eval_requests_after_training.assert_not_called()
    test_get_eval_requests_after_pipeline.assert_called_once()
    test_launch_evaluations_async.assert_called_once()
    assert evaluation_executor.context.dataset_end_time == 99


@patch.object(EvaluationExecutor, "_single_batched_evaluation", return_value=[("failure", {}), ("failure", {})])
def test_launch_evaluations_async(
    test_single_evaluation: Any, evaluation_executor: EvaluationExecutor, dummy_stage_log: StageLog
) -> None:
    evaluation_executor._launch_evaluations_async(
        eval_requests=[dummy_eval_request(), dummy_eval_request()],
        parent_log=dummy_stage_log,
    )
    assert test_single_evaluation.call_count == 1  # batched


@patch.object(GRPCHandler, "get_evaluation_results")
@patch.object(GRPCHandler, "wait_for_evaluation_completion")
@patch.object(GRPCHandler, "prepare_evaluation_request")
def test_single_batched_evaluation_all_failed(
    test_prepare_evaluation_request: Any,
    test_wait_for_evaluation_completion: Any,
    test_get_evaluation_results: Any,
    evaluation_executor: EvaluationExecutor,
) -> None:
    evaluator_stub_mock = mock.Mock(spec=["evaluate_model"])
    evaluator_stub_mock.evaluate_model.return_value = EvaluateModelResponse(
        evaluation_started=False,
        interval_responses=[
            EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.DATASET_NOT_FOUND),
            EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.EMPTY_DATASET),
        ],
    )

    evaluation_executor.grpc.evaluator = evaluator_stub_mock
    eval_req = dummy_eval_request()
    eval_req2 = dummy_eval_request()
    eval_req2.interval_start = 63
    eval_req2.interval_end = 14
    results = evaluation_executor._single_batched_evaluation(
        [
            (eval_req.interval_start, eval_req.interval_end),
            (eval_req2.interval_start, eval_req2.interval_end),
        ],
        eval_req.id_model,
        eval_req.dataset_id,
    )
    assert results == [("DATASET_NOT_FOUND", {}), ("EMPTY_DATASET", {})]

    test_prepare_evaluation_request.assert_called_once_with(
        ANY, eval_req.id_model, ANY, intervals=[(0, None), (63, 14)]
    )
    test_wait_for_evaluation_completion.assert_not_called()
    test_get_evaluation_results.assert_not_called()


@patch.object(GRPCHandler, "cleanup_evaluations")
@patch.object(
    GRPCHandler,
    "get_evaluation_results",
    return_value=[
        EvaluationIntervalData(
            interval_index=1,
            evaluation_data=[SingleMetricResult(metric="accuracy", result=0.5)],
        )
    ],
)
@patch.object(GRPCHandler, "wait_for_evaluation_completion")
@patch.object(GRPCHandler, "prepare_evaluation_request")
def test_single_batched_evaluation_mixed(
    test_prepare_evaluation_request: Any,
    test_wait_for_evaluation_completion: Any,
    test_get_evaluation_results: Any,
    test_cleanup_evaluations: Any,
    evaluation_executor: EvaluationExecutor,
) -> None:
    evaluator_stub_mock = mock.Mock(spec=["evaluate_model"])

    evaluator_stub_mock.evaluate_model.return_value = EvaluateModelResponse(
        evaluation_started=True,
        evaluation_id=42,
        interval_responses=[
            EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.EMPTY_DATASET),
            EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.NOT_ABORTED, dataset_size=21),
        ],
    )

    evaluation_executor.grpc.evaluator = evaluator_stub_mock
    eval_req = dummy_eval_request()
    eval_req2 = dummy_eval_request()
    eval_req.interval_start = 64
    eval_req.interval_end = 14

    request = MagicMock()
    request.metrics = ["Accuracy"]
    test_prepare_evaluation_request.return_value = request

    results = evaluation_executor._single_batched_evaluation(
        [
            (eval_req.interval_start, eval_req.interval_end),
            (eval_req2.interval_start, eval_req2.interval_end),
        ],
        eval_req.id_model,
        eval_req.dataset_id,
    )
    assert results == [
        ("EMPTY_DATASET", {}),
        (None, {"dataset_size": 21, "metrics": [{"name": "accuracy", "result": pytest.approx(0.5)}]}),
    ]

    test_prepare_evaluation_request.assert_called_once_with(
        ANY, eval_req.id_model, ANY, intervals=[(64, 14), (0, None)]
    )
    test_wait_for_evaluation_completion.assert_called_once()
    test_get_evaluation_results.assert_called_once()
    assert test_cleanup_evaluations.call_count == 2
