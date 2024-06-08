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
from modyn.config.schema.pipeline import (
    AfterTrainingEvalTriggerConfig,
    EvalDataConfig,
    EvalHandlerConfig,
    MatrixEvalTriggerConfig,
    ModynPipelineConfig,
    PeriodicEvalTriggerConfig,
    UntilNextTriggerEvalStrategyConfig,
)
from modyn.config.schema.pipeline.evaluation.trigger.static import StaticEvalTriggerConfig
from modyn.config.schema.system import ModynConfig
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluateModelResponse, EvaluationAbortedReason
from modyn.supervisor.internal.eval.handler import EvalHandler
from modyn.supervisor.internal.eval.result_writer.json_result_writer import JsonResultWriter
from modyn.supervisor.internal.eval.strategies.abstract_eval_strategy import AbstractEvalStrategy
from modyn.supervisor.internal.eval.triggers.after_training_trigger import AfterTrainingTrigger
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalRequest, EvalTrigger
from modyn.supervisor.internal.eval.triggers.periodic_eval_trigger import PeriodicEvalTrigger
from modyn.supervisor.internal.eval.triggers.static_eval_trigger import StaticEvalTrigger
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
class EvalContext:
    tracking_dfs: dict[str, pd.DataFrame]


class EvaluationExecutor:
    def __init__(
        self, pipeline_id: int, eval_dir: Path, config: ModynConfig, pipeline: ModynPipelineConfig, grpc: GRPCHandler
    ):
        self.pipeline_id = pipeline_id
        self.eval_dir = eval_dir
        self.config = config
        self.pipeline = pipeline
        self.grpc = grpc
        self.context: EvalContext | None = None

    def register_tracking_info(self, tracking_dfs: dict[str, pd.DataFrame]) -> None:
        assert tracking_dfs.get(PipelineStage.HANDLE_SINGLE_TRIGGER.name) is not None
        assert tracking_dfs.get(PipelineStage.STORE_TRAINED_MODEL.name) is not None
        self.context = EvalContext(tracking_dfs=tracking_dfs)

    def create_snapshot(self, snapshot_dir: Path) -> None:
        """Create a snapshot of the evaluation state before starting to evaluate."""
        if not self.pipeline.evaluation:
            return

        # create tempdir if snapshot_dir is None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # write state: config, pipeline & context
        eval_state_config = EvalStateConfig(
            pipeline_id=self.pipeline_id, eval_dir=self.eval_dir, config=self.config, pipeline=self.pipeline
        )
        (snapshot_dir / "eval_state.yaml").write_text(eval_state_config.model_dump_json(by_alias=True))
        (snapshot_dir / "context.pcl").write_bytes(pickle.dumps(self.context))

    @classmethod
    def init_from_path(cls, snapshot_dir: Path) -> "EvaluationExecutor":
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

        logs: SupervisorLogs = SupervisorLogs()

        for eval_handler in self.pipeline.evaluation.handlers:
            if not isinstance(eval_handler.trigger, AfterTrainingEvalTriggerConfig) or (
                eval_handler.strategy.type == "UntilNextTriggerEvalStrategy"
            ):
                # those cases are handled via the _post_pipeline_evaluation setup
                continue

            eval_strategy: AbstractEvalStrategy = getattr(eval_strategy_module, eval_handler.strategy.type)(
                eval_handler.strategy
            )

            for eval_dataset_config in [
                self.pipeline.evaluation.datasets[dataset_ref] for dataset_ref in eval_handler.datasets
            ]:
                for interval_start, interval_end in eval_strategy.get_eval_intervals(first_timestamp, last_timestamp):
                    eval_req = EvalRequest(
                        trigger_id=trigger_id,
                        training_id=training_id,
                        model_id=model_id,
                        most_recent_model=True,
                        interval_start=interval_start,
                        interval_end=interval_end,
                    )

                    single_log = StageLog(
                        id="EVALUATE_SINGLE",
                        start=datetime.datetime.now(),
                        batch_idx=log.batch_idx,
                        sample_idx=log.sample_idx,
                        sample_time=log.sample_time,
                        trigger_idx=log.trigger_idx,
                    )
                    epoch_nanos_start = current_time_micros()
                    self._single_evaluation(
                        single_log, eval_status_queue, eval_req, eval_handler.name, eval_dataset_config
                    )
                    single_log.end = datetime.datetime.now()
                    single_log.duration = datetime.timedelta(microseconds=current_time_micros() - epoch_nanos_start)
                    logs.stage_runs.append(single_log)

        return logs

    def run_post_pipeline_evaluations(self) -> SupervisorLogs:  # pylint: disable=too-many-locals
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

        tasks: list[Future] = []
        eval_requests: list[EvalRequest] = []
        logs = SupervisorLogs()

        # The time measurements for evaluation requests is negligible, let's parallelization to speed up evaluation;
        # As we are io bound by the evaluator server, GIL locking isn't a concerns. This enables multithreading.
        num_threads = 10
        if self.config.supervisor:
            num_threads = self.config.supervisor.post_pipeline_evaluation_workers
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for eval_handler_config in self.pipeline.evaluation.handlers:
                if isinstance(eval_handler_config.trigger, AfterTrainingEvalTriggerConfig) and (
                    not isinstance(eval_handler_config.strategy, UntilNextTriggerEvalStrategyConfig)
                ):
                    continue  # handled core pipeline

                eval_handler = _eval_handler_factory(eval_handler_config)
                if not eval_handler.trigger:
                    continue

                for eval_dataset_config in [
                    self.pipeline.evaluation.datasets[dataset_ref] for dataset_ref in eval_handler_config.datasets
                ]:
                    if isinstance(eval_handler_config.strategy, UntilNextTriggerEvalStrategyConfig):
                        assert isinstance(eval_handler_config.trigger, AfterTrainingEvalTriggerConfig)

                        # we have the model and interval values already in the AfterTrainingTrigger
                        eval_requests = eval_handler.trigger.get_eval_requests(df_trainings)

                    else:
                        # we have the trigger invocation timestamp in the EvalTrigger
                        # however, we need to match those to trigger_ids and model_ids
                        build_matrix = (
                            eval_handler_config.trigger.matrix
                            if isinstance(eval_handler_config.trigger, MatrixEvalTriggerConfig)
                            else False
                        )
                        raw_eval_requests = eval_handler.trigger.get_eval_requests(df_trainings, build_matrix)

                        eval_strategy: AbstractEvalStrategy = getattr(
                            eval_strategy_module, eval_handler_config.strategy.type
                        )(eval_handler_config.strategy)

                        eval_requests = [
                            EvalRequest(
                                trigger_id=raw_eval_req.trigger_id,
                                training_id=raw_eval_req.model_id,
                                model_id=raw_eval_req.model_id,
                                most_recent_model=raw_eval_req.most_recent_model,
                                interval_start=interval_start,
                                interval_end=interval_end,
                            )
                            for raw_eval_req in raw_eval_requests
                            for interval_start, interval_end in (
                                eval_strategy.get_eval_intervals(
                                    raw_eval_req.interval_start, raw_eval_req.interval_end  # type: ignore
                                )
                            )
                        ]

                    def eval_thread_job(
                        eval_req: EvalRequest, dataset_config: EvalDataConfig, eval_handler_name: str
                    ) -> None:
                        log = StageLog(
                            id="POST_PIPELINE_EVALUATION",
                            start=datetime.datetime.now(),
                            end=None,
                            batch_idx=-1,
                            sample_idx=-1,
                            sample_time=-1,
                            trigger_idx=-1,
                        )
                        epoch_nanos_start = current_time_micros()
                        self._single_evaluation(log, Queue(), eval_req, eval_handler_name, dataset_config)
                        log.duration = datetime.timedelta(microseconds=current_time_micros() - epoch_nanos_start)
                        logs.stage_runs.append(log)

                    for eval_req in eval_requests:
                        task = partial(eval_thread_job, eval_req, eval_dataset_config, eval_handler_config.name)
                        tasks.append(pool.submit(task))

            # join threads
            for t in tasks:
                t.result()

        return logs

    # -------------------------------------------------------------------------------- #
    #                                     Internal                                     #
    # -------------------------------------------------------------------------------- #

    def _single_evaluation(
        self,
        log: StageLog,
        eval_status_queue: Queue,
        eval_request: EvalRequest,
        eval_handler: str,
        eval_dataset_config: EvalDataConfig,
    ) -> None:
        assert self.grpc.evaluator is not None, "Evaluator not initialized."
        assert self.pipeline.evaluation
        logger.info(
            f"Evaluation Starts for model {eval_request.model_id} on split {eval_request.interval_start}"
            f" to {eval_request.interval_end}"
            f" of dataset {eval_dataset_config.dataset_id}."
        )
        request = GRPCHandler.prepare_evaluation_request(
            eval_dataset_config.model_dump(by_alias=True),
            eval_request.model_id,
            self.pipeline.evaluation.device,
            eval_request.interval_start,
            eval_request.interval_end,
        )
        response: EvaluateModelResponse = self.grpc.evaluator.evaluate_model(request)
        log.info = SingleEvaluationInfo(
            trigger_id=eval_request.trigger_id,
            id_model=eval_request.model_id,
            eval_handler=eval_handler,
            most_recent_model=eval_request.most_recent_model,
            dataset_id=eval_dataset_config.dataset_id,
            interval_start=eval_request.interval_start,
            interval_end=eval_request.interval_end,
        )
        if not response.evaluation_started:
            log.info.failure_reason = EvaluationAbortedReason.DESCRIPTOR.values_by_number[
                response.eval_aborted_reason
            ].name
            logger.error(
                f"Evaluation for model {eval_request.model_id} on split {eval_request.interval_start} to"
                f" {eval_request.interval_end} not started with reason: {log.info.failure_reason}."
            )
            return

        logger.info(
            f"Evaluation started for model {eval_request.model_id} on split {eval_request.interval_start}"
            f" to {eval_request.interval_end}."
        )
        reporter = EvaluationStatusReporter(
            eval_status_queue,
            response.evaluation_id,
            eval_dataset_config.dataset_id,
            response.dataset_size,
        )
        evaluation = {response.evaluation_id: reporter}
        reporter.create_tracker()
        self.grpc.wait_for_evaluation_completion(eval_request.training_id, evaluation)

        eval_result_writer = JsonResultWriter(self.pipeline_id, eval_request.trigger_id, self.eval_dir)
        self.grpc.store_evaluation_results([eval_result_writer], evaluation)
        self.grpc.cleanup_evaluations([int(i) for i in evaluation])
        assert isinstance(eval_result_writer, JsonResultWriter)

        log.info.results = (
            eval_result_writer.results["datasets"][0][eval_dataset_config.dataset_id]
            if eval_result_writer.results["datasets"]
            else {}
        )


# ------------------------------------------------------------------------------------ #
#                                       Internal                                       #
# ------------------------------------------------------------------------------------ #


def _eval_handler_factory(eval_handler_configs: EvalHandlerConfig) -> EvalHandler:
    if isinstance(eval_handler_configs.trigger, AfterTrainingEvalTriggerConfig):
        trigger: EvalTrigger = AfterTrainingTrigger()
    elif isinstance(eval_handler_configs.trigger, StaticEvalTriggerConfig):
        trigger = StaticEvalTrigger(eval_handler_configs.trigger)
    elif isinstance(eval_handler_configs.trigger, PeriodicEvalTriggerConfig):
        trigger = PeriodicEvalTrigger(eval_handler_configs.trigger)
    else:
        raise ValueError(f"Unknown evaluation trigger mode: {eval_handler_configs.trigger.mode}")
    return EvalHandler(config=eval_handler_configs, trigger=trigger)


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

    logs_.supervisor_logs = ex.run_post_pipeline_evaluations()
    logs_.materialize(snapshot_path, mode="final")
