"""Coordinates the evaluation after the core pipeline execution."""

import datetime
import logging
import pickle
import sys
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import grpc
import pandas as pd
from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.config.schema.system import ModynConfig

# pylint: disable-next=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluateModelResponse, EvaluationAbortedReason
from modyn.supervisor.internal.eval.handler import EvalHandler, EvalRequest
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.pipeline_executor.models import (
    ConfigLogs,
    MultiEvaluationInfo,
    PipelineLogs,
    SingleEvaluationInfo,
    StageLog,
    SupervisorLogs,
)
from modyn.utils.utils import current_time_micros, dynamic_module_import
from pydantic import BaseModel
from tenacity import Retrying, stop_after_attempt, wait_random_exponential

eval_strategy_module = dynamic_module_import("modyn.supervisor.internal.eval.strategies")

logger = logging.getLogger(__name__)


class EvalStateConfig(BaseModel):
    pipeline_id: int
    eval_dir: Path
    config: ModynConfig
    pipeline: ModynPipelineConfig


@dataclass
class AfterPipelineEvalContext:
    tracking_dfs: dict[str, pd.DataFrame]


class EvaluationExecutor:
    def __init__(
        self,
        pipeline_id: int,
        pipeline_logdir: Path,
        config: ModynConfig,
        pipeline: ModynPipelineConfig,
        grpc_handler: GRPCHandler,
    ):
        self.pipeline_id = pipeline_id
        self.pipeline_logdir = pipeline_logdir
        self.config = config
        self.pipeline = pipeline
        self.grpc = grpc_handler
        self.context: AfterPipelineEvalContext | None = None
        self.eval_handlers = (
            [EvalHandler(eval_handler_config) for eval_handler_config in pipeline.evaluation.handlers]
            if pipeline.evaluation
            else []
        )

    def register_tracking_info(self, tracking_dfs: dict[str, pd.DataFrame]) -> None:
        assert tracking_dfs.get(PipelineStage.HANDLE_SINGLE_TRIGGER.name) is not None
        assert tracking_dfs.get(PipelineStage.STORE_TRAINED_MODEL.name) is not None
        self.context = AfterPipelineEvalContext(tracking_dfs=tracking_dfs)

    def create_snapshot(self) -> None:
        """Create a snapshot of the pipeline metadata before starting to evaluate."""
        if not self.pipeline.evaluation:
            return

        snapshot_dir = self.pipeline_logdir / "snapshot"

        # create tempdir if snapshot_dir is None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # write state: config, pipeline & context
        eval_state_config = EvalStateConfig(
            pipeline_id=self.pipeline_id, eval_dir=self.pipeline_logdir, config=self.config, pipeline=self.pipeline
        )
        (snapshot_dir / "eval_state.yaml").write_text(eval_state_config.model_dump_json(by_alias=True))
        (snapshot_dir / "context.pcl").write_bytes(pickle.dumps(self.context))

    @classmethod
    def init_from_path(cls, pipeline_logdir: Path) -> "EvaluationExecutor":
        snapshot_dir = pipeline_logdir / "snapshot"
        assert snapshot_dir.exists()

        # read state: config, pipeline & context
        eval_state_config = EvalStateConfig.model_validate_json((snapshot_dir / "eval_state.yaml").read_text())
        context = pickle.loads((snapshot_dir / "context.pcl").read_bytes())

        grpc_handler = GRPCHandler(eval_state_config.config.model_dump(by_alias=True))
        executor = EvaluationExecutor(
            eval_state_config.pipeline_id,
            eval_state_config.eval_dir,
            eval_state_config.config,
            eval_state_config.pipeline,
            grpc_handler,
        )
        executor.grpc.init_cluster_connection()
        executor.context = context
        return executor

    def run_pipeline_evaluations(
        self,
        log: StageLog,
        trigger_id: int,
        training_id: int,
        model_id: int,
        first_timestamp: int,
        last_timestamp: int,
    ) -> SupervisorLogs:
        """Run the evaluations as part of the core pipeline.

        Args:
            log: The stage log of the caller pipeline stage.
            pipeline_status_queue: The queue to communicate the pipeline status.
            trigger_id: The trigger id to evaluate.
            training_id: The training id to evaluate.
            model_id: The model id to evaluate.
            first_timestamp: Start of the training interval.
            last_timestamp: End of the training interval.

        Returns:
            The logs of the evaluations.
        """
        assert self.grpc.evaluator is not None, "Evaluator not initialized."
        assert self.pipeline.evaluation is not None, "Evaluation config not set."

        eval_requests: list[EvalRequest] = []

        for eval_handler in self.eval_handlers:
            if eval_handler.config.execution_time != "after_training":
                continue

            handler_eval_requests = eval_handler.get_eval_requests_after_training(
                trigger_id=trigger_id,
                training_id=training_id,
                model_id=model_id,
                training_interval=(first_timestamp, last_timestamp),
            )
            eval_requests += handler_eval_requests

        if len(eval_requests) == 0:
            return SupervisorLogs()
        num_workers = self.pipeline.evaluation.after_training_evaluation_workers
        logs = self._launch_evaluations_async(eval_requests, log, num_workers)
        return logs

    def run_post_pipeline_evaluations(self, manual_run: bool = False) -> SupervisorLogs:
        """Evaluate the trained models after the core pipeline and store the results."""
        if not self.pipeline.evaluation:
            return SupervisorLogs(stage_runs=[])

        assert self.context, "EvaluationExecutor not initialized"

        df_triggers = self.context.tracking_dfs.get(PipelineStage.HANDLE_SINGLE_TRIGGER.name)
        df_store_models = self.context.tracking_dfs.get(PipelineStage.STORE_TRAINED_MODEL.name)
        if df_triggers is None or df_store_models is None:
            logger.warning("Could not run evaluations as no training was found in the tracking info dataframes")
            return SupervisorLogs(stage_runs=[])
        df_trainings = df_triggers[["trigger_id", "first_timestamp", "last_timestamp"]].merge(
            df_store_models[["id_model", "training_id", "trigger_id"]], on="trigger_id"
        )

        eval_requests: list[EvalRequest] = []
        for eval_handler in self.eval_handlers:
            if (eval_handler.config.execution_time not in ("after_pipeline", "manual")) or (
                eval_handler.config.execution_time == "manual" and not manual_run
            ):
                continue

            handler_eval_requests = eval_handler.get_eval_requests_after_pipeline(df_trainings=df_trainings)
            eval_requests += handler_eval_requests

        if len(eval_requests) == 0:
            return SupervisorLogs()

        # self.Eval_handlers is not an empty list if and only if self.pipeline.evaluation is not None
        assert self.pipeline.evaluation is not None

        logs = self._launch_evaluations_async(
            eval_requests,
            # as we don't execute this during the training pipeline, we don't have a reference how
            # our current process is in terms of position in the dataset.
            parent_log=StageLog(
                id=PipelineStage.EVALUATE_MULTI.name,
                id_seq_num=-1,
                start=datetime.datetime.now(),
                batch_idx=-1,
                sample_idx=-1,
                sample_time=-1,
                trigger_idx=-1,
            ),
            num_workers=self.pipeline.evaluation.after_pipeline_evaluation_workers,
        )
        return logs

    # -------------------------------------------------------------------------------- #
    #                                     Internal                                     #
    # -------------------------------------------------------------------------------- #

    def _launch_evaluations_async(
        self,
        eval_requests: list[EvalRequest],
        parent_log: StageLog,
        num_workers: int = 1,
    ) -> SupervisorLogs:
        """Creates a thread pool to launch evaluations asynchronously.

        Args:
            eval_requests: The evaluation requests to launch.
            num_workers: The number of workers to use.
        """

        tasks: list[Future[StageLog]] = []
        logs = SupervisorLogs()

        # do batching by (dataset_id, model_id)
        eval_requests_by_model: dict[tuple[str, int], list[EvalRequest]] = defaultdict(list)
        for eval_req in eval_requests:
            key_ = (eval_req.dataset_id, eval_req.id_model)
            eval_requests_by_model[key_] += [eval_req]

        def worker_func(model_eval_req: tuple[tuple[str, int], list[EvalRequest]]) -> StageLog:
            """
            Args:
                model_eval_req: A tuple of model_id and a list of evaluation requests for that model.
            """
            dataset_id = model_eval_req[0][0]
            model_id = model_eval_req[0][1]
            eval_requests = model_eval_req[1]
            intervals = [(eval_req.interval_start, eval_req.interval_end) for eval_req in eval_requests]

            epoch_micros_start = current_time_micros()

            # intervals and results are in the same order
            results = self._single_batched_evaluation(intervals, model_id, dataset_id)

            interval_result_infos: list[SingleEvaluationInfo] = []
            intervals_and_results = zip(eval_requests, results)
            for eval_req, (failure_reason, interval_res) in intervals_and_results:
                interval_result_infos.append(
                    SingleEvaluationInfo(
                        eval_request=eval_req,
                        failure_reason=failure_reason if failure_reason else None,
                        results=interval_res if not failure_reason else {},
                    )
                )

            single_log = StageLog(
                id=PipelineStage.EVALUATE_MULTI.name,
                id_seq_num=-1,  # evaluation don't need sequence numbers, their order is not important
                start=datetime.datetime.now(),
                batch_idx=parent_log.batch_idx,
                sample_idx=parent_log.sample_idx,
                sample_time=parent_log.sample_time,
                trigger_idx=parent_log.trigger_idx,
                info=None,
            )
            single_log.info = MultiEvaluationInfo(
                dataset_id=dataset_id,
                id_model=model_id,
                interval_results=interval_result_infos,
            )
            single_log.end = datetime.datetime.now()
            single_log.duration = datetime.timedelta(microseconds=current_time_micros() - epoch_micros_start)
            return single_log

        # As we are io bound by the evaluator server, GIL locking isn't a concern, so we can use multithreading.
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for r in eval_requests_by_model.items():
                task = partial(worker_func, r)
                tasks.append(pool.submit(task))

            # join threads
            for t in tasks:
                logs.stage_runs.append(t.result())

        return logs

    # pylint: disable-next=too-many-locals
    def _single_batched_evaluation(
        self,
        intervals: list[tuple[int, int | None]],
        model_id_to_eval: int,
        dataset_id: str,
    ) -> list[tuple[str | None, dict]]:
        """Takes a list of intervals to be evaluated for a certain model and dataset.

        Returns:
            A list of tuples, where the first element is the failure reason (if any) and the second element is the
            evaluation results. The order of the tuples corresponds to the order of the intervals.
        """
        assert self.grpc.evaluator is not None, "Evaluator not initialized."
        assert self.pipeline.evaluation
        logger.info(
            f"Evaluation Starts for model {model_id_to_eval} and dataset {dataset_id} on intervals:  {intervals}."
        )
        dataset_config = next((d for d in self.pipeline.evaluation.datasets if d.dataset_id == dataset_id))

        request = GRPCHandler.prepare_evaluation_request(
            dataset_config.model_dump(by_alias=True),
            model_id_to_eval,
            self.pipeline.evaluation.device,
            intervals=cast(list[tuple[int | None, int | None]], intervals),
        )
        for attempt in Retrying(
            stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, min=2, max=60), reraise=True
        ):
            with attempt:
                try:
                    response: EvaluateModelResponse = self.grpc.evaluator.evaluate_model(request)
                except grpc.RpcError as e:  # We catch and reraise to reconnect
                    logger.error(e)
                    logger.error("gRPC connection error, trying to reconnect...")
                    self.grpc.init_evaluator()
                    raise e

        assert len(response.interval_responses) == len(
            intervals
        ), f"We expected {len(intervals)} intervals, but got {len(response.interval_responses)}."

        def get_failure_reason(eval_aborted_reason: EvaluationAbortedReason) -> str:
            return EvaluationAbortedReason.DESCRIPTOR.values_by_number[eval_aborted_reason].name

        if not response.evaluation_started:
            failure_reasons: list[tuple[str | None, dict]] = []
            # note: interval indexes correspond to the intervals in the request
            for interval_idx, interval_response in enumerate(response.interval_responses):
                if interval_response.eval_aborted_reason != EvaluationAbortedReason.NOT_ABORTED:
                    reason = get_failure_reason(interval_response.eval_aborted_reason)
                    failure_reasons.append((reason, {}))
                    logger.error(
                        f"Evaluation for model {model_id_to_eval} on split {intervals[interval_idx]} "
                        f"not started with reason: {reason}."
                    )
            return failure_reasons

        logger.info(f"Evaluation started for model {model_id_to_eval} on intervals {intervals}.")
        self.grpc.wait_for_evaluation_completion(response.evaluation_id)
        eval_data = self.grpc.get_evaluation_results(response.evaluation_id)
        self.grpc.cleanup_evaluations([response.evaluation_id])

        eval_results: list[tuple[str | None, dict[str, Any]]] = []

        # ---------------------------------------------- Result Builder ---------------------------------------------- #
        # The `eval_results` list is a list of tuples. Each tuple contains a failure reason (if any) and a dictionary
        # with the evaluation results. The order of the tuples corresponds to the order of the intervals.
        #
        # response.interval_responses contains the evaluation results for each interval in the same order as the
        # intervals in the request. Failed evaluations are marked with a failure reason.

        # Metric results come from the `EvaluateModelResponse` and are stored in the `evaluation_data` field. This
        # only contains the metrics for the intervals that were successfully evaluated.
        #
        # Therefore we first build a list of results with the same order as the intervals. The metrics will be filled in
        # the next loop that unwraps `EvaluationResultResponse`.
        # ----------------------------------------------------- . ---------------------------------------------------- #

        for interval_response in response.interval_responses:
            if interval_response.eval_aborted_reason != EvaluationAbortedReason.NOT_ABORTED:
                reason = get_failure_reason(interval_response.eval_aborted_reason)
                eval_results.append((reason, {}))
            else:
                eval_results.append((None, {"dataset_size": interval_response.dataset_size, "metrics": []}))

        for interval_result in eval_data:
            interval_idx = interval_result.interval_index
            assert eval_results[interval_idx][0] is None, "Evaluation failed, no metrics should be present."
            eval_results[interval_idx][1]["metrics"] = [
                {"name": metric.metric, "result": metric.result} for metric in interval_result.evaluation_data
            ]
        return eval_results


# ------------------------------------------------------------------------------------ #
#                                       DevTools                                       #
# ------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    snapshot_path = Path(input("Enter pipeline log directory path to (re)run evaluation executor: "))
    if not snapshot_path.exists():
        print("Path not found")
        sys.exit(1)

    # restart evaluation executor
    ex = EvaluationExecutor.init_from_path(snapshot_path)

    logs_ = PipelineLogs(
        pipeline_id=ex.pipeline_id,
        pipeline_stages={},
        config=ConfigLogs(system=ex.config, pipeline=ex.pipeline),
        experiment=True,
        supervisor_logs=SupervisorLogs(),
    )

    logs_.supervisor_logs = ex.run_post_pipeline_evaluations()
    logs_.materialize(snapshot_path, mode="final")
    logger.info("Done with manual evaluation!")
    sys.exit(0)
