"""Coordinates the evaluation after the core pipeline execution."""

import datetime
import logging
import pickle
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import Queue
from pathlib import Path

import pandas as pd
from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.config.schema.system import ModynConfig
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluateModelResponse, EvaluationAbortedReason
from modyn.supervisor.internal.eval.handler import EvalHandler, EvalRequest
from modyn.supervisor.internal.eval.result_writer.json_result_writer import JsonResultWriter
from modyn.supervisor.internal.grpc.enums import IdType, MsgType, PipelineStage
from modyn.supervisor.internal.grpc.template_msg import id_submsg, pipeline_stage_msg
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.pipeline_executor.models import (
    ConfigLogs,
    PipelineLogs,
    SingleEvaluationInfo,
    StageLog,
    SupervisorLogs,
)
from modyn.supervisor.internal.utils.evaluation_status_reporter import EvaluationStatusReporter
from modyn.utils.utils import current_time_micros, dynamic_module_import
from pydantic import BaseModel

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
        grpc: GRPCHandler,
    ):
        self.pipeline_id = pipeline_id
        self.pipeline_logdir = pipeline_logdir
        self.config = config
        self.pipeline = pipeline
        self.grpc = grpc
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
        """Create a snapshot of the evaluation state before starting to evaluate."""
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

        executor = EvaluationExecutor(
            eval_state_config.pipeline_id,
            eval_state_config.eval_dir,
            eval_state_config.config,
            eval_state_config.pipeline,
            GRPCHandler(eval_state_config.config.model_dump(by_alias=True)),
        )
        executor.context = context
        return executor

    # eval_requests: list[EvalRequest] = []

    # for eval_handler in self.eval_handlers:
    #     if eval_handler.config.execution_time != "after_training":
    #         continue

    #     handler_eval_requests = eval_handler.get_eval_requests_after_training(
    #         trigger_id=trigger_id,
    #         training_id=training_id,
    #         model_id=model_id,
    #         training_interval=(first_timestamp, last_timestamp),
    #     )
    #     eval_requests += handler_eval_requests

    # # will be replaced with threadpool in a follow-up PR
    # for eval_req in eval_requests:
    #     results = self._single_evaluation(s, self.logs, eval_req)
    #     if results:
    #         self._store_evaluation_results(s, self.logs, eval_req, results)

    def run_pipeline_evaluations(
        self,
        log: StageLog,
        trigger_id: int,
        training_id: int,
        model_id: int,
        first_timestamp: int,
        last_timestamp: int,
        pipeline_status_queue: Queue,
        eval_status_queue: Queue,
    ) -> SupervisorLogs:
        """Run the evaluations as part of the core pipeline.

        Args:
            log: The stage log of the caller pipeline stage.
            pipeline_status_queue: The queue to communicate the pipeline status.
            trigger_id: The trigger id to evaluate.
            training_id: The training id to evaluate.
            model_id: The model id to evaluate.
            first_timestamp: Start of the interval to evaluate.
            last_timestamp: End of the interval to evaluate.

        Returns:
            The logs of the evaluations.
        """
        # pylint: disable=too-many-locals
        assert self.grpc.evaluator is not None, "Evaluator not initialized."
        assert self.pipeline.evaluation is not None, "Evaluation config not set."
        pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.EVALUATE, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )

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

        num_workers = None
        if self.config.supervisor:
            num_workers = self.config.supervisor.after_training_evaluation_workers
        logs = self._launch_evaluations_async(eval_requests, log, eval_status_queue, num_workers)
        return logs

    def run_post_pipeline_evaluations(
        self, eval_status_queue: Queue
    ) -> SupervisorLogs:  # pylint: disable=too-many-locals
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
            if eval_handler.config.execution_time != "after_pipeline":
                continue

            handler_eval_requests = eval_handler.get_eval_requests_after_pipeline(df_trainings=df_trainings)
            eval_requests += handler_eval_requests

        num_workers = None
        if self.config.supervisor:
            num_workers = self.config.supervisor.after_pipeline_evaluation_workers

        logs = self._launch_evaluations_async(
            eval_requests,
            parent_log=StageLog(
                id=PipelineStage.EVALUATE_SINGLE.name,
                start=datetime.datetime.now(),
                batch_idx=-1,
                sample_idx=-1,
                sample_time=-1,
                trigger_idx=-1,
            ),
            eval_status_queue=eval_status_queue,
            num_workers=num_workers,
        )
        return logs

    # -------------------------------------------------------------------------------- #
    #                                     Internal                                     #
    # -------------------------------------------------------------------------------- #

    def _launch_evaluations_async(
        self,
        eval_requests: list[EvalRequest],
        parent_log: StageLog,
        eval_status_queue: Queue,
        num_workers: int | None = None,
    ) -> SupervisorLogs:
        """Creates a thread pool to launch evaluations asynchronously.

        Args:
            eval_requests: The evaluation requests to launch.
            num_workers: The number of workers to use.
        """

        tasks: list[Future[StageLog]] = []
        logs = SupervisorLogs()

        def worker_func(eval_req: EvalRequest) -> StageLog:
            single_log = StageLog(
                id=PipelineStage.EVALUATE_SINGLE.name,
                start=datetime.datetime.now(),
                batch_idx=parent_log.batch_idx,
                sample_idx=parent_log.sample_idx,
                sample_time=parent_log.sample_time,
                trigger_idx=parent_log.trigger_idx,
            )
            epoch_nanos_start = current_time_micros()
            self._single_evaluation(single_log, eval_status_queue, eval_req)
            single_log.end = datetime.datetime.now()
            single_log.duration = datetime.timedelta(microseconds=current_time_micros() - epoch_nanos_start)
            return single_log

        # As we are io bound by the evaluator server, GIL locking isn't a concern, so we can use multithreading.
        with ThreadPoolExecutor(max_workers=num_workers or 1) as pool:
            for eval_req in eval_requests:
                task = partial(worker_func, eval_req)
                tasks.append(pool.submit(task))

            # join threads
            for t in tasks:
                logs.stage_runs.append(t.result())

        return logs

    def _single_evaluation(self, log: StageLog, eval_status_queue: Queue, eval_req: EvalRequest) -> None:
        assert self.grpc.evaluator is not None, "Evaluator not initialized."
        assert self.pipeline.evaluation
        logger.info(
            f"Evaluation Starts for model {eval_req.model_id} on split {eval_req.interval_start} "
            f"to {eval_req.interval_end} of dataset {eval_req.dataset_id}."
        )
        dataset_config = next((d for d in self.pipeline.evaluation.datasets if d.dataset_id == eval_req.dataset_id))
        log.info = SingleEvaluationInfo(eval_request=eval_req)
        request = GRPCHandler.prepare_evaluation_request(
            dataset_config.model_dump(by_alias=True),
            eval_req.model_id,
            self.pipeline.evaluation.device,
            eval_req.interval_start,
            eval_req.interval_end,
        )
        response: EvaluateModelResponse = self.grpc.evaluator.evaluate_model(request)
        if not response.evaluation_started:
            log.info.failure_reason = EvaluationAbortedReason.DESCRIPTOR.values_by_number[
                response.eval_aborted_reason
            ].name
            logger.error(
                f"Evaluation for model {eval_req.model_id} on split {eval_req.interval_start} to "
                f"{eval_req.interval_end} not started with reason: {log.info.failure_reason}."
            )
            return

        logger.info(
            f"Evaluation started for model {eval_req.model_id} on split {eval_req.interval_start} "
            f"to {eval_req.interval_end}."
        )
        reporter = EvaluationStatusReporter(
            eval_status_queue,
            response.evaluation_id,
            eval_req.dataset_id,
            response.dataset_size,
        )
        evaluation = {response.evaluation_id: reporter}
        reporter.create_tracker()
        self.grpc.wait_for_evaluation_completion(eval_req.training_id, evaluation)

        eval_result_writer = JsonResultWriter(self.pipeline_id, eval_req.trigger_id, self.pipeline_logdir)
        self.grpc.store_evaluation_results([eval_result_writer], evaluation)
        self.grpc.cleanup_evaluations([int(i) for i in evaluation])
        assert isinstance(eval_result_writer, JsonResultWriter)

        log.info.results = (
            eval_result_writer.results["datasets"][0][dataset_config.dataset_id]
            if eval_result_writer.results["datasets"]
            else {}
        )


# ------------------------------------------------------------------------------------ #
#                                       DevTools                                       #
# ------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    snapshot_path = Path(input("Enter eval snapshot path to (re)run evaluation executor: "))
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

    logs_.supervisor_logs = ex.run_post_pipeline_evaluations(eval_status_queue=Queue())
    logs_.materialize(snapshot_path, mode="final")
