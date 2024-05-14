# pylint: disable=unused-argument

from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime
from time import sleep
from typing import Any, Callable

import pandas as pd

from modyn.supervisor.internal.evaluation_result_writer import LogResultWriter
from modyn.supervisor.internal.grpc.enums import (
    CounterAction,
    IdType,
    MsgType,
    PipelineStage,
)
from modyn.supervisor.internal.grpc.template_msg import (
    counter_submsg,
    dataset_submsg,
    id_submsg,
    pipeline_stage_msg,
)
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.triggers import Trigger
from modyn.utils import dynamic_module_import

from .models import (
    ConfigLogs,
    EvaluateTriggerInfo,
    EvaluationInfo,
    ExecutionState,
    HandleNewDataInfo,
    PipelineBatchState,
    PipelineLogs,
    PipelineOptions,
    RegisteredStage,
    SelectorInformLog,
    SelectorInformTriggerLog,
    StageLog,
    StoreModelInfo,
    TrainingInfo,
)

logger = logging.getLogger(__name__)
EXCEPTION_EXITCODE = 8

PipelineRegistry = dict[PipelineStage, RegisteredStage]

# Dynamically registered pipeline stages (via decorators)
# for subpipeline hierarchy, see `PIPELINE.md`
main_pipeline: PipelineRegistry = {}
replay_data_pipeline: PipelineRegistry = {}
wait_for_new_data_pipeline: PipelineRegistry = {}
new_data_pipeline: PipelineRegistry = {}
new_data_batch_pipeline: PipelineRegistry = {}
execute_trigger: PipelineRegistry = {}
training_and_eval_pipeline: PipelineRegistry = {}
training_pipeline: PipelineRegistry = {}
evaluation_pipeline: PipelineRegistry = {}


def register_stage(
    pipeline: dict[PipelineStage, RegisteredStage],
    stage: PipelineStage,
    *,
    then: PipelineStage | None = None,
    log: bool = True,
    track: bool = False,
) -> Callable:
    """Decorator to register a pipeline stage handler function.

    Args:
        pipeline: The pipeline to register the stage in.
        stage: The stage to register.
        then: The next stage to execute after the current stage; returns values of the stage function override this.
        log: Whether to log the stage execution.
        track: Whether to track the stage execution and make results available in the pipeline (e.g. trigger policies).
    """

    def wrapper_outer(func: Callable[..., PipelineStage | None]) -> Callable[..., PipelineStage | None]:
        assert stage not in pipeline
        pipeline[stage] = RegisteredStage(stage=stage, func=func, next=then, log=log, track=track)

        def wrapper(*args: Any, **kwargs: Any) -> PipelineStage | None:
            return func(*args, **kwargs)  # type: ignore

        return wrapper

    return wrapper_outer


class PipelineExecutor:
    def __init__(self, options: PipelineOptions) -> None:
        self.stage = PipelineStage.INIT
        self.state = ExecutionState(**vars(options))
        self.logs = PipelineLogs(
            pipeline_id=options.pipeline_id,
            config=ConfigLogs(system=options.modyn_config, pipeline=options.pipeline_config),
            experiment=options.experiment_mode,
            start_replay_at=options.start_replay_at,
            stop_replay_at=options.stop_replay_at,
        )
        """Execution state of the pipeline executor."""

        # pipeline controllers objects
        self.trigger = self._setup_trigger()
        self.grpc = GRPCHandler(
            self.state.modyn_config,
            self.state.pipeline_status_queue,
            self.state.training_status_queue,
            self.state.eval_status_queue,
        )

    # -------------------------------------------- Pipeline orchestration -------------------------------------------- #

    def execute(
        self, pipeline: dict[PipelineStage, RegisteredStage], initial_stage: PipelineStage = PipelineStage.INIT
    ) -> None:
        """Coordinates all pipelines stages until the pipeline execution is finished.

        Measures the time for each stage and logs the pipeline state.
        """
        self.stage = initial_stage
        while True:
            stage = pipeline[self.stage]

            if stage.log:
                logger.info(f"[pipeline {self.state.pipeline_id}] Entering <{stage.stage}>.")

            # execute stage
            log = StageLog(
                id=stage.stage.name,
                start=datetime.now(),
                sample_idx=self.state.current_sample_idx,
                sample_time=self.state.current_sample_time,
            )
            returned_stage = stage.func(self, log)
            log.end = datetime.now()

            # if stage reported additional logs, we make the log available to the pipeline in a dataframe
            if stage.track and log.info:
                # ensure df exists
                old_df = self.state.tracking.get(log.id, None)
                if new_rows := log.online_df():
                    self.state.tracking[log.id] = pd.concat(old_df, new_rows) if old_df else new_rows

            # record logs
            if stage.log:
                self.logs.supervisor.stage_runs.append(log)
                logger.info(f"[pipeline {self.state.pipeline_id}] Finished <{stage.stage}>.")

            # state transition
            if returned_stage and stage.next:
                raise RuntimeError("A pipeline stage must either pre-define a next stage or return a stage. Not both!")
            if stage.next:
                self.stage = stage.next
            elif returned_stage:
                self.stage = returned_stage
            else:
                break

    # ------------------------------------------------ Pipeline stages ----------------------------------------------- #

    # These functions are not suppose to be called manually.

    # Setup

    @register_stage(main_pipeline, PipelineStage.INIT, then=PipelineStage.INIT_CLUSTER_CONNECTION, log=False)
    def _init(self, log: StageLog) -> None:
        self.state.new_data.max_timestamp = self.state.start_timestamp
        self.logs.materialize(self.state.log_directory, mode="initial")
        if self.state.pipeline_config.training.initial_model == "pretrained":
            self.state.previous_model_id = self.state.pipeline_config.training.initial_model_id

    @register_stage(main_pipeline, PipelineStage.INIT_CLUSTER_CONNECTION, then=PipelineStage._FORK_DATA_STRATEGY)
    def _init_cluster_connection(self, log: StageLog) -> None:
        self.state.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.INIT_CLUSTER_CONNECTION, MsgType.GENERAL))
        self.grpc.init_cluster_connection()

    @register_stage(main_pipeline, PipelineStage._FORK_DATA_STRATEGY, then=PipelineStage.DONE, log=False)
    def _fork_data_strategy(self, log: StageLog) -> None:
        """Run either the `replay_data_pipeline` or the `wait_for_new_data_pipeline` subpipelines."""

        if self.state.experiment_mode:
            logger.info(f"Running pipeline {self.state.pipeline_id} in experiment mode.")
            self.execute(replay_data_pipeline, initial_stage=PipelineStage.REPLAY_DATA)
            logger.info(f"Experiment mode pipeline {self.state.pipeline_id} done.")
        else:
            logger.info(f"Running pipeline {self.state.pipeline_id} in new data mode.")
            logger.info("Press CTRL+C at any time to shutdown the pipeline.")
            try:
                self.execute(wait_for_new_data_pipeline, initial_stage=PipelineStage.FETCH_NEW_DATA)
            except KeyboardInterrupt:
                logger.info("Initiating shutdown.")
                self._shutdown_trainer()
                logger.info("Shutdown successful.")
            logger.info(f"New data mode pipeline {self.state.pipeline_id} done.")

    # Replay Data (experiment mode)

    @register_stage(replay_data_pipeline, PipelineStage.REPLAY_DATA, then=PipelineStage.REPLAY_DATA_DONE)
    def _replay_data(self, log: StageLog) -> None:
        assert self.state.start_replay_at is not None, "Cannot call replay_data when start_replay_at is None"
        dataset_id = self.state.pipeline_config.data.dataset_id
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.REPLAY_DATA, MsgType.DATASET, dataset_submsg(dataset_id))
        )
        logger.info("Starting data replay.")

        if self.state.stop_replay_at is None:
            replay_data_generator = self.grpc.get_new_data_since(dataset_id, self.state.start_replay_at)
        else:
            replay_data_generator = self.grpc.get_data_in_interval(
                dataset_id, self.state.start_replay_at, self.state.stop_replay_at
            )

        for self.state.new_data, self.state.new_data.fetch_time in replay_data_generator:
            # update pipeline current_sample_idx and current_sample_time
            self.state.current_sample_time = min(
                (timestamp for (_, timestamp, _) in self.state.new_data), default=self.state.current_sample_time
            )

            # Run new data subpipeline
            self.execute(new_data_pipeline, PipelineStage.HANDLE_NEW_DATA)

            if self.state.maximum_triggers is not None and self.state.num_triggers >= self.state.maximum_triggers:
                logger.info("Exiting replay loop due to trigger limit.")
                break

    @register_stage(replay_data_pipeline, PipelineStage.REPLAY_DATA_DONE, then=None)
    def _replay_data_done(self, log: StageLog) -> None:
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.REPLAY_DATA_DONE,
                MsgType.DATASET,
                dataset_submsg(self.state.pipeline_config.data.dataset_id),
            )
        )

        # finish replay_data subpipeline

    # Wait for new data (non-experiment mode)

    @register_stage(wait_for_new_data_pipeline, PipelineStage.FETCH_NEW_DATA, then=None)
    def _fetch_new_data(self, log: StageLog) -> PipelineStage | None:
        dataset_id = self.state.pipeline_config.data.dataset_id
        continue_running = True

        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.FETCH_NEW_DATA, MsgType.DATASET, dataset_submsg(dataset_id))
        )

        trigger_occurred = False
        largest_keys = set()
        replay_data_generator = self.grpc.get_new_data_since(dataset_id, self.state.new_data.max_timestamp)

        for self.state.new_data, self.state.new_data.fetch_time in replay_data_generator:
            # Since get_new_data_since is inclusive, we have to expect more yet unprocessed samples
            # with `new_data_max_timestamp` alongside already processed samples with `new_data_max_timestamp`.
            # We memorize the already processed samples to avoid processing them again. Using set to get O(1) lookup.
            self.state.new_data = [
                (key, timestamp, label)
                for (key, timestamp, label) in self.state.new_data
                if key not in self.state.previous_largest_keys
            ]
            self.state.new_data.max_timestamp = (
                max((timestamp for (_, timestamp, _) in self.state.new_data))
                if len(self.state.new_data) > 0
                else self.state.new_data.max_timestamp
            )
            largest_keys.update(
                {key for (key, timestamp, _) in self.state.new_data if timestamp == self.state.new_data.max_timestamp}
            )

            # update pipeline current_sample_idx and current_sample_time
            self.state.current_sample_time = min(
                (timestamp for (_, timestamp, _) in self.state.new_data), default=self.state.current_sample_time
            )

            # process batch: writes new_data_had_trigger
            self.execute(new_data_pipeline, PipelineStage.HANDLE_NEW_DATA)

            if self.state.new_data.had_trigger:
                trigger_occurred = True

            if self.state.maximum_triggers is not None and self.state.num_triggers >= self.state.maximum_triggers:
                continue_running = False

        self.state.previous_largest_keys = largest_keys

        if not trigger_occurred and continue_running:
            return PipelineStage.WAIT_FOR_NEW_DATA  # another iteration

        return None  # finish wait_for_new_data subpipeline

    @register_stage(wait_for_new_data_pipeline, PipelineStage.WAIT_FOR_NEW_DATA, then=PipelineStage.FETCH_NEW_DATA)
    def _wait_for_new_data(self, log: StageLog) -> None:
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.WAIT_FOR_NEW_DATA,
                MsgType.DATASET,
                dataset_submsg(self.state.pipeline_config.data.dataset_id),
            )
        )
        sleep(2)

    # Process new data

    @register_stage(new_data_pipeline, PipelineStage.HANDLE_NEW_DATA, then=PipelineStage.NEW_DATA_HANDLED, track=True)
    def _handle_new_data(self, log: StageLog) -> None:
        """Handle new data during experiments or actual pipeline execution.

        We partition `new_data` into batches of `selector_batch_size` to reduce selector latency in case of a trigger.
        If a data point within a batch causes a trigger,
        we inform the selector about all data points including that data point.
        Otherwise, the selector is informed.
        # TODO: evaluate if this batching strategy is still necessary or if it can be emulated
        #  by trigger strategies returning multiple indices
        """
        self.state.new_data.data.sort(key=lambda tup: tup[1])  # sort by timestamp
        any_training_triggered = False
        new_data_len = len(self.state.new_data.data)

        logger.info(f"Received {new_data_len} new data points. Handling batches.")
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.HANDLE_NEW_DATA,
                MsgType.COUNTER,
                counter_submsg(CounterAction.CREATE, {"new_data_len": new_data_len}),
            )
        )

        for i in range(0, new_data_len, self.state.selector_batch_size):
            batch = self.state.new_data.data[i : i + self.state.selector_batch_size]
            batch_size = (
                self.state.selector_batch_size
                if i + self.state.selector_batch_size < new_data_len
                else new_data_len - i
            )
            if batch_size > 0:
                self.state.current_sample_time = batch[0]  # update sample time
            self.state.pipeline_status_queue.put(
                pipeline_stage_msg(
                    PipelineStage.HANDLE_NEW_DATA,
                    MsgType.COUNTER,
                    counter_submsg(CounterAction.UPDATE, {"batch_size": batch_size}),
                )
            )

            # execute new batch subpipeline, writes previous_batch_had_trigger
            self.state.batch = PipelineBatchState(data=batch, remaining_data=batch)
            self.execute(new_data_batch_pipeline, PipelineStage.EVALUATE_TRIGGER_ON_BATCH)
            any_training_triggered = any_training_triggered or self.state.previous_batch_had_trigger

            # update sample counter
            self.state.current_sample_idx += batch_size

            if self.state.maximum_triggers is not None and self.state.num_triggers >= self.state.maximum_triggers:
                logger.info(f"Reached trigger limit ({self.state.maximum_triggers}), exiting.")
                break

        self.state.new_data.had_trigger = any_training_triggered

        # log extra information
        log.info = HandleNewDataInfo(
            fetch_time=self.state.new_data.fetch_time, num_samples=new_data_len, had_trigger=any_training_triggered
        )

    @register_stage(new_data_pipeline, PipelineStage.NEW_DATA_HANDLED, then=None)
    def _new_data_handled(self, log: StageLog) -> None:
        self.state.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.NEW_DATA_HANDLED, MsgType.GENERAL))
        self.logs.materialize(self.state.log_directory, mode="intermediate")

        # end of `new_data_pipeline`

    # Process new data batch

    @register_stage(
        new_data_batch_pipeline, 
        PipelineStage.EVALUATE_TRIGGER_ON_BATCH, 
        then=PipelineStage.EXECUTE_TRIGGERS_WITHIN_BATCH, 
        track=True
    )
    def _evaluate_trigger_policies(self, log: StageLog) -> PipelineStage:
        """Evaluate trigger policy and inform selector."""

        # Evaluate trigger policy
        triggering_indices = self.trigger.inform(self.state.new_data.data)

        num_triggers = len(triggering_indices)

        # persist state
        self.state.num_triggers = num_triggers
        self.state.previous_batch_had_trigger = num_triggers > 0
        self.state.batch.triggering_indices = triggering_indices

        # add log data
        log.info = EvaluateTriggerInfo(num_triggers=num_triggers, triggering_indices=triggering_indices)

    @register_stage(new_data_batch_pipeline, PipelineStage.EXECUTE_TRIGGERS_WITHIN_BATCH, track=True)
    def _execute_triggers_within_batch(self, log: StageLog) -> PipelineStage | None:  # [useless-return]
        """Evaluate trigger policy, start training after trigger and inform selector."""
        logger.info(f"There are {self.state.num_triggers} triggers in this batch.")
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.EXECUTE_TRIGGERS_WITHIN_BATCH, MsgType.GENERAL)
        )

        logger.info("Handling triggers within batch.")
        triggering_idx = -1
        for triggering_idx in self.state.batch.triggering_indices:
            self.state.batch.trigger_index = triggering_idx

            # Run training and evaluation subpipelines
            self.execute(execute_trigger, PipelineStage.INFORM_SELECTOR_AND_TRIGGER)

            self.state.num_triggers = self.state.num_triggers + 1
            if self.state.maximum_triggers is not None and self.state.num_triggers >= self.state.maximum_triggers:
                break

        # If no other trigger is coming in this batch, we  inform the Selector about the remaining data in this batch.
        if triggering_idx > -1:
            self.state.batch.remaining_data = self.state.new_data.data[triggering_idx + 1 :]

        logger.info(f"There are {len(self.state.batch.remaining_data)} data points remaining after the trigger.")

        if len(self.state.batch.remaining_data) > 0:
            if self.state.remaining_data_range is not None:
                # extend the range from last time
                self.state.remaining_data_range = (
                    self.state.remaining_data_range[0], self.state.batch.remaining_data[-1][1]
                )
            else:
                self.state.remaining_data_range = (
                    self.state.batch.remaining_data[0][1], self.state.remaining_data_range[-1][1]
                )
            return PipelineStage.INFORM_SELECTOR_REMAINING_DATA

        return None  # End of `new_data_batch_pipeline` subpipeline

    @register_stage(new_data_pipeline, PipelineStage.INFORM_SELECTOR_REMAINING_DATA, then=None, track=True)
    def _inform_selector_remaining_data(self, log: StageLog) -> None:
        """Inform selector about remaining data."""
        
        # These data points will be included in the next trigger because we inform the Selector about them,
        # just like other batches with no trigger at all are included.
        selector_log = self.grpc.inform_selector(self.state.pipeline_id, self.state.batch.remaining_data)

        # add log data
        log.info = SelectorInformLog(selector_log=selector_log, seen_trigger=self.state.batch.previous_trigger_idx)

        # end of `new_data_pipeline`


    # Execute trigger within batch

    @register_stage(
        execute_trigger, PipelineStage.INFORM_SELECTOR_AND_TRIGGER, then=PipelineStage.TRAIN_AND_EVALUATE, track=True
    )
    def _inform_selector_and_trigger(self, log: StageLog) -> None:
        # unpack state
        batch = self.state.new_data.data
        trigger_idx = self.state.batch.trigger_index

        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.INFORM_SELECTOR_AND_TRIGGER, MsgType.GENERAL)
        )
        triggering_data = batch[self.state.batch.previous_trigger_idx : trigger_idx + 1]
        first_timestamp, last_timestamp = self._get_trigger_timespan(
            self.state.batch.previous_trigger_idx == 0, triggering_data
        )
        self.state.batch.previous_trigger_idx = self.state.batch.trigger_index + 1

        # This call informs the selector about the data until (and including) the data point that caused the trigger
        # and then also notifies it about the triggering. This means the next training call on trigger_id will
        # guarantee that all data until that point has been processed by the selector.
        self.state.batch.trigger_id, selector_log = self.grpc.inform_selector_and_trigger(
            self.state.pipeline_id, triggering_data
        )

        # add log data
        log.info = SelectorInformTriggerLog(
            trigger_index=trigger_idx,
            trigger_id=self.state.batch.trigger_id,
            selector_log=selector_log,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
        )

    @register_stage(execute_trigger, PipelineStage.TRAIN_AND_EVALUATE, track=True)
    def _run_train_eval_pipeline(self, log: StageLog) -> None:
        num_samples_in_trigger = self.grpc.get_number_of_samples(self.state.pipeline_id, self.state.batch.trigger_index)
        if num_samples_in_trigger > 0:
            # Blocks until training is done.
            self.execute(training_pipeline, PipelineStage.RUN_TRAINING)

            self.state.pipeline_status_queue.put(
                pipeline_stage_msg(
                    PipelineStage.EXECUTE_TRIGGERS_WITHIN_BATCH,
                    MsgType.ID,
                    id_submsg(IdType.TRIGGER, self.state.batch.trigger_id),
                )
            )

            self.execute(evaluation_pipeline, PipelineStage.EVALUATE)
            self.state.remaining_data_range = None

        else:
            logger.info(f"Skipping training on empty trigger {self.state.batch.trigger_index}]")

    # Training

    @register_stage(training_pipeline, PipelineStage.RUN_TRAINING, then=PipelineStage.TRAINING_COMPLETED, track=True)
    def _run_training(self, log: StageLog) -> None:
        """Run training for trigger on GPU and block until done."""

        logger.info(f"Running training for trigger {self.state.batch.trigger_id}")
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.RUN_TRAINING, MsgType.ID, id_submsg(IdType.TRIGGER, self.state.batch.trigger_id)
            )
        )
        first_timestamp, last_timestamp = self._get_trigger_timespan(i == 0, triggering_data)

        num_samples_to_pass_per_trigger = self.state.pipeline_config.training.num_samples_to_pass or []
        current_trigger_index = len(self.state.triggers)
        if current_trigger_index <= len(num_samples_to_pass_per_trigger) - 1:
            num_samples_to_pass = num_samples_to_pass_per_trigger[current_trigger_index]
        else:
            num_samples_to_pass = None

        self.state.training_id = self.grpc.start_training(
            self.state.pipeline_id,
            self.state.batch.trigger_id,
            self.state.pipeline_config,
            self.state.previous_model_id,
            num_samples_to_pass,
        )

        trainer_log = self.grpc.wait_for_training_completion(
            self.state.training_id, self.state.pipeline_id, self.state.batch.trigger_index
        )

        # add log data
        log.info = TrainingInfo(
            trigger_index=self.state.batch.trigger_index,
            trigger_id=self.state.batch.trigger_id,
            training_id=self.state.training_id,
            num_samples_to_pass=num_samples_to_pass,
            trainer_log=trainer_log,
        )

    @register_stage(training_pipeline, PipelineStage.TRAINING_COMPLETED, then=PipelineStage.STORE_TRAINED_MODEL)
    def _training_completed(self, log: StageLog) -> None:
        assert self.state.training_id
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.TRAINING_COMPLETED, MsgType.ID, id_submsg(IdType.TRAINING, self.state.training_id), True
            )
        )
        logger.info(f"Training {self.state.training_id} completed")

    @register_stage(training_pipeline, PipelineStage.STORE_TRAINED_MODEL, then=None, track=True)
    def _store_trained_model(self, log: StageLog) -> None:
        assert self.state.training_id
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.STORE_TRAINED_MODEL, MsgType.ID, id_submsg(IdType.TRIGGER, self.state.batch.trigger_id)
            )
        )

        # We store the trained model for evaluation in any case.
        model_id = self.grpc.store_trained_model(self.state.training_id)

        # Only if the pipeline actually wants to continue the training on it, we set previous model.
        if self.state.pipeline_config.training.use_previous_model:
            self.state.previous_model_id = model_id

        self.state.trained_models.append(model_id)
        self.state.triggers.append(self.state.batch.trigger_index)

        # add log data
        log.info = StoreModelInfo(
            trigger_index=self.state.batch.trigger_index,
            trigger_id=self.state.batch.trigger_id,
            training_id=self.state.training_id,
            model_id=model_id,
        )

        # End of training subpipeline

    # Evaluation

    @register_stage(evaluation_pipeline, PipelineStage.EVALUATE, then=PipelineStage.EVALUATION_COMPLETED, track=True)
    def _evaluate(self, log: StageLog) -> None:
        model_id = self.state.trained_models[-1]

        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.EVALUATE, MsgType.ID, id_submsg(IdType.TRIGGER, self.state.batch.trigger_index)
            )
        )

        self.state.batch.evaluations = self.grpc.start_evaluation(model_id, self.state.pipeline_config)
        self.grpc.wait_for_evaluation_completion(self.state.training_id, self.state.batch.evaluations)

        # add log data
        log.info = EvaluationInfo(
            trigger_index=self.state.batch.trigger_index,
            trigger_id=self.state.batch.trigger_id,
            training_id=self.state.training_id,
            model_id=model_id,
        )

    @register_stage(
        evaluation_pipeline, PipelineStage.EVALUATION_COMPLETED, then=PipelineStage.STORE_EVALUATION_RESULTS
    )
    def _evaluation_completed(self, log: StageLog) -> None:
        pass  # nothing to do

    @register_stage(
        evaluation_pipeline,
        PipelineStage.STORE_EVALUATION_RESULTS,
        then=PipelineStage.EXECUTE_TRIGGERS_WITHIN_BATCH,
        track=True,
    )
    def _store_evaluation_results(self, log: StageLog) -> None:
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.STORE_EVALUATION_RESULTS,
                MsgType.ID,
                id_submsg(IdType.TRIGGER, self.state.batch.trigger_id),
            )
        )

        writer_names: set[str] = set(self.state.pipeline_config.evaluation.result_writers)
        writers = [self._init_evaluation_writer(name, self.state.batch.trigger_id) for name in writer_names]
        self.grpc.store_evaluation_results(writers, self.state.batch.evaluations)

    # Teardown

    @register_stage(main_pipeline, PipelineStage.DONE, then=PipelineStage.EXIT)
    def _done(self, log: StageLog) -> None:
        self.state.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.DONE, MsgType.GENERAL))
        self.logs.materialize(self.state.log_directory, mode="final")

    @register_stage(main_pipeline, PipelineStage.EXIT, then=None)
    def _exit(self, log: StageLog) -> None:
        return None  # end of pipeline

    # ---------------------------------------------------- Helpers --------------------------------------------------- #

    # setup

    def _setup_trigger(self) -> Trigger:
        trigger_id = self.state.pipeline_config.trigger.id
        trigger_config = self.state.pipeline_config.trigger.trigger_config

        trigger_module = dynamic_module_import("modyn.supervisor.internal.triggers")
        trigger: Trigger = getattr(trigger_module, trigger_id)(trigger_config)
        assert trigger is not None, "Error during trigger initialization"

        return trigger

    # pipeline run
    
    def _get_trigger_timespan(
        self, is_first_triggering_data: bool, triggering_data: list[tuple[int, int, int]]
    ) -> tuple[int, int]:
        if is_first_triggering_data:
            # now it is the first trigger in this batch. Triggering_data can be empty.
            # when it is indeed empty, then there is remaining data in the last batch
            # because num_samples_in_trigger is not 0.
            assert len(triggering_data) > 0 or self.state.remaining_data_range is not None

            if self.state.remaining_data_range is not None:
                first_timestamp = self.state.remaining_data_range[0]
                last_timestamp = (
                    self.state.remaining_data_range[1]
                    if len(triggering_data) == 0
                    else triggering_data[-1][1]
                )
            else:
                first_timestamp = triggering_data[0][1]
                last_timestamp = triggering_data[-1][1]
        else:
            assert len(triggering_data) > 0
            # since num_samples_in_trigger is not 0, we are sure that triggering_data is not empty
            first_timestamp = triggering_data[0][1]
            last_timestamp = triggering_data[-1][1]

        return first_timestamp, last_timestamp

    def _init_evaluation_writer(self, name: str, trigger_id: int) -> LogResultWriter:
        return self.state.supervisor_supported_eval_result_writers[name](
            self.state.pipeline_id, trigger_id, self.state.eval_directory
        )

    def _shutdown_trainer(self) -> None:
        if self.state.training_id is not None:
            self.grpc.stop_training_at_trainer_server(self.state.training_id)


def execute_pipeline(options: PipelineOptions) -> None:
    try:
        logger.info(f"[pipeline {options.pipeline_id}] Start executing, experiment mode {options.experiment_mode}.")
        PipelineExecutor(options).execute(main_pipeline)
        logger.info(f"[pipeline {options.pipeline_id}] Execution done. Persist log.")

    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        options.exception_queue.put(exception_msg)
        sys.exit(EXCEPTION_EXITCODE)
