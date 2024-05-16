# pylint: disable=unused-argument
from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime
from time import sleep
from typing import Callable, Generator, TypeVar

import pandas as pd
from modyn.supervisor.internal.evaluation_result_writer import LogResultWriter
from modyn.supervisor.internal.grpc.enums import CounterAction, IdType, MsgType, PipelineStage, PipelineType
from modyn.supervisor.internal.grpc.template_msg import counter_submsg, dataset_submsg, id_submsg, pipeline_stage_msg
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.triggers import Trigger
from modyn.supervisor.internal.utils.evaluation_status_reporter import EvaluationStatusReporter
from modyn.utils import dynamic_module_import
from typing_extensions import Concatenate, ParamSpec

from .models import (
    ConfigLogs,
    EvaluateTriggerInfo,
    EvaluationInfo,
    ExecutionState,
    FetchDataInfo,
    PipelineLogs,
    PipelineOptions,
    ProcessNewDataInfo,
    SelectorInformLog,
    SelectorInformTriggerInfo,
    StageLog,
    StoreModelInfo,
    TrainingInfo,
    TriggerExecutionInfo,
)

logger = logging.getLogger(__name__)
EXCEPTION_EXITCODE = 8

P = ParamSpec("P")  # parameters of pipeline stage
R = TypeVar("R")  # result of pipeline stage


def pipeline_stage(  # type: ignore[no-untyped-def]
    pipeline: PipelineType,
    stage: PipelineStage,
    *,
    log: bool = True,
    track: bool = False,
):
    """Decorator to register a pipeline stage handler function.

    Args:
        pipeline: The pipeline to register the stage in.
        stage: The stage to register.
        then: The next stage to execute after the current stage
        log: Whether to log the stage execution.
        track: Whether to track the stage execution and make results available in the pipeline (e.g. trigger policies).
    """

    def wrapper_outer(  # type: ignore[no-untyped-def]
        func: Callable[Concatenate["PipelineExecutor", ExecutionState, StageLog, P], R]
    ):
        def wrapper(
            self: "PipelineExecutor", state: ExecutionState, logs: PipelineLogs, *args: P.args, **kwargs: P.kwargs
        ) -> R:
            """Measures the time for each stage and logs the pipeline state."""
            state.stage = stage
            if log:
                logger.info(f"[pipeline {pipeline}: {state.pipeline_id}] Entering <{stage}>.")

            # execute stage
            stage_log = StageLog(
                id=stage.name,
                start=datetime.now(),
                sample_idx=state.current_sample_index,
                sample_time=state.current_sample_time,
            )
            result = func(self, state, stage_log, *args, **kwargs)  # type: ignore[call-arg]
            stage_log.end = datetime.now()
            state.stage = stage  # restore stage as child pipeline might have changed it

            # if stage reported additional logs, we make the log available to the pipeline in a dataframe
            if track and stage_log.info:
                # ensure df exists
                old_df = state.tracking.get(stage_log.id, None)
                if (new_rows := stage_log.online_df(extended=True)) is not None:
                    state.tracking[stage_log.id] = pd.concat([old_df, new_rows]) if old_df is not None else new_rows

            # record logs
            if log:
                logs.supervisor.stage_runs.append(stage_log)
                logger.info(f"[pipeline {state.pipeline_id}] Finished <{stage.name}>.")

            # result of stage function
            return result

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
            self.state.modyn_config.model_dump(by_alias=True),
            self.state.pipeline_status_queue,
            self.state.training_status_queue,
            self.state.eval_status_queue,
        )

    def run(self) -> None:
        """Execute the main pipeline."""
        logger.info(
            f"[pipeline {self.state.pipeline_id}] Start executing, experiment mode {self.state.experiment_mode}."
        )

        self._init(self.state, self.logs)
        self._init_cluster_connection(self.state, self.logs)

        if self.state.experiment_mode:
            self._replay_data(self.state, self.logs)
        else:
            self._serve_online_data(self.state, self.logs)

        self._done(self.state, self.logs)
        self._exit(self.state, self.logs)

        logger.info(f"[pipeline {self.state.pipeline_id}] Execution done. Persist log.")

    # ------------------------------------------------ Pipeline stages ----------------------------------------------- #

    # Setup

    @pipeline_stage(PipelineType.MAIN, PipelineStage.INIT, log=False)
    def _init(self, s: ExecutionState, log: StageLog) -> None:
        s.max_timestamp = s.start_timestamp
        self.logs.materialize(s.log_directory, mode="initial")
        if s.pipeline_config.training.initial_model == "pretrained":
            s.previous_model_id = s.pipeline_config.training.initial_model_id

    @pipeline_stage(PipelineType.MAIN, PipelineStage.INIT_CLUSTER_CONNECTION)
    def _init_cluster_connection(self, s: ExecutionState, log: StageLog) -> None:
        s.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.INIT_CLUSTER_CONNECTION, MsgType.GENERAL))
        self.grpc.init_cluster_connection()

    # Replay Data (experiment mode)

    @pipeline_stage(PipelineType.REPLAY_DATA, PipelineStage.REPLAY_DATA)
    def _replay_data(self, s: ExecutionState, log: StageLog) -> None:
        assert s.start_replay_at is not None, "Cannot call replay_data when start_replay_at is None"

        logger.info(f"Running pipeline {s.pipeline_id} in experiment mode.")
        logger.info("Starting data replay.")
        s.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.REPLAY_DATA, MsgType.DATASET, dataset_submsg(s.dataset_id))
        )

        if s.stop_replay_at is None:
            replay_data_generator = self.grpc.get_new_data_since(s.dataset_id, s.start_replay_at)
        else:
            replay_data_generator = self.grpc.get_data_in_interval(s.dataset_id, s.start_replay_at, s.stop_replay_at)

        for replay_data, request_time in replay_data_generator:
            # setting sample here to have correct logs in process_new_data
            s.current_sample_time = min(
                (timestamp for (_, timestamp, _) in replay_data),
                default=s.start_timestamp,
            )
            s.current_sample_index = replay_data[0][0] if replay_data else 0

            self._process_new_data(s, self.logs, replay_data, request_time)

            if s.maximum_triggers is not None and len(s.triggers) >= s.maximum_triggers:
                logger.info("Exiting replay loop due to trigger limit.")
                break

        logger.info(f"Experiment mode pipeline {s.pipeline_id} done.")
        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.REPLAY_DATA_DONE,
                MsgType.DATASET,
                dataset_submsg(s.dataset_id),
            )
        )

    # Online serving mode (non-experiment mode)

    @pipeline_stage(PipelineType.SERVE_ONLINE, PipelineStage.SERVE_ONLINE_DATA)
    def _serve_online_data(self, s: ExecutionState, log: StageLog) -> None:
        """Run pipeline in production mode fetching new data until pipeline is stopped."""
        logger.info(f"Running pipeline {s.pipeline_id} in online serving mode.")
        logger.info("Press CTRL+C at any time to shutdown the pipeline.")

        try:
            while True:
                num_new_triggers = self._fetch_new_data(s, self.logs)
                if s.maximum_triggers_reached:
                    break
                if num_new_triggers == 0:
                    self._wait_for_new_data(s, self.logs)

        except KeyboardInterrupt:
            logger.info("Initiating shutdown.")
            self._shutdown_trainer()
            logger.info("Shutdown successful.")

        logger.info(f"New data mode pipeline {s.pipeline_id} done.")

    @pipeline_stage(PipelineType.SERVE_ONLINE, PipelineStage.FETCH_NEW_DATA, track=True)
    def _fetch_new_data(self, s: ExecutionState, log: StageLog) -> int:
        """Try to fetch new data from the dataset and process it.

        Returns:
            The number of triggers that occurred during the processing of the new data.
        """
        s.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.FETCH_NEW_DATA, MsgType.DATASET, dataset_submsg(s.dataset_id))
        )

        num_samples = 0
        trigger_indexes: list[int] = []
        largest_keys = set()
        replay_data_generator = self.grpc.get_new_data_since(s.dataset_id, s.max_timestamp)

        for fetched_data, fetch_time in replay_data_generator:
            # Since get_new_data_since is inclusive, we have to expect more yet unprocessed samples
            # with `max_timestamp` alongside already processed samples with `max_timestamp`.
            # We memorize the already processed samples to avoid processing them again. Using set to get O(1) lookup.
            fetched_data = [
                (key, timestamp, label)
                for (key, timestamp, label) in fetched_data
                if key not in s.previous_largest_keys
            ]
            s.max_timestamp = (
                max((timestamp for (_, timestamp, _) in fetched_data)) if len(fetched_data) > 0 else s.max_timestamp
            )
            largest_keys.update({key for (key, timestamp, _) in fetched_data if timestamp == s.max_timestamp})

            # setting sample here to have correct logs in process_new_data
            s.current_sample_time = min(
                (timestamp for (_, timestamp, _) in fetched_data),
                default=s.start_timestamp,
            )
            s.current_sample_index = fetched_data[0][0] if fetched_data else 0

            # process new data and invoke triggers
            trigger_indexes = trigger_indexes + self._process_new_data(s, self.logs, fetched_data, fetch_time)
            num_samples += len(fetched_data)

        s.previous_largest_keys = largest_keys

        # log extra information
        log.info = FetchDataInfo(num_samples=num_samples, trigger_indexes=trigger_indexes)

        return len(trigger_indexes)

    @pipeline_stage(PipelineType.SERVE_ONLINE, PipelineStage.WAIT_FOR_NEW_DATA)
    def _wait_for_new_data(self, s: ExecutionState, log: StageLog) -> None:
        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.WAIT_FOR_NEW_DATA,
                MsgType.DATASET,
                dataset_submsg(s.dataset_id),
            )
        )
        sleep(2)

    # Process new data

    @pipeline_stage(PipelineType.NEW_DATA, PipelineStage.PROCESS_NEW_DATA, track=True)
    def _process_new_data(
        self, s: ExecutionState, log: StageLog, new_data: list[tuple[int, int, int]], fetch_time: int
    ) -> list[int]:
        """Handle new data during experiments or online pipeline serving.

        We partition `new_data` into batches of `selector_batch_size` to reduce selector latency in case of a trigger.
        If a data point within a batch causes a trigger, we inform the selector about all data points including
        that data point. Otherwise, the selector is informed.

        Args:
            s: Execution state of the pipeline executor.
            log: Log of the current stage.
            new_data: List of tuples (key, timestamp, label) of new data points.
            fetch_time: Number of milliseconds it took to fetch the data.

        Returns:
            List of indexes of data points that caused a trigger.
        """
        new_data.sort(key=lambda tup: tup[1])  # sort by timestamp
        trigger_indexes: list[int] = []
        new_data_len = len(new_data)

        logger.info(f"Received {new_data_len} new data points. Handling batches.")
        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.PROCESS_NEW_DATA,
                MsgType.COUNTER,
                counter_submsg(CounterAction.CREATE, {"new_data_len": new_data_len}),
            )
        )

        for i in range(0, new_data_len, s.selector_batch_size):
            batch = new_data[i : i + s.selector_batch_size]
            batch_size = s.selector_batch_size if i + s.selector_batch_size < new_data_len else new_data_len - i
            if batch_size > 0:
                s.current_sample_index = batch[0][0]  # update sample index
                s.current_sample_time = batch[0][1]  # update sample time

            s.pipeline_status_queue.put(
                pipeline_stage_msg(
                    PipelineStage.PROCESS_NEW_DATA,
                    MsgType.COUNTER,
                    counter_submsg(CounterAction.UPDATE, {"batch_size": batch_size}),
                )
            )

            trigger_indexes += self._process_new_data_batch(s, self.logs, batch)

            if s.maximum_triggers is not None and len(s.triggers) >= s.maximum_triggers:
                logger.info(f"Reached trigger limit ({s.maximum_triggers}), exiting.")
                break

        s.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.NEW_DATA_HANDLED, MsgType.COUNTER, counter_submsg(CounterAction.CLOSE))
        )

        # log extra information
        log.info = ProcessNewDataInfo(fetch_time=fetch_time, num_samples=new_data_len, trigger_indexes=trigger_indexes)
        self.logs.materialize(s.log_directory, mode="increment")

        return trigger_indexes

    # Process new data BATCH

    @pipeline_stage(PipelineType.NEW_BATCH, PipelineStage.PROCESS_NEW_DATA_BATCH, track=True)
    def _process_new_data_batch(self, s: ExecutionState, log: StageLog, batch: list[tuple[int, int, int]]) -> list[int]:
        """Process new data in batches and evaluate trigger policies in batches.

        Args:
            s: Execution state of the pipeline executor.
            log: Log of the current stage.
            batch: List of tuples (key, timestamp, label) of new data points.

        Returns:
            List of indexes of data points that caused a trigger.
        """

        # Evaluate trigger policy and inform selector
        lazy_trigger_indexes = self._evaluate_trigger_policies(s, self.logs, batch)

        # Execute triggers within batch (training & evaluation subpipelines)
        executed_triggers = self._execute_triggers(s, self.logs, batch, lazy_trigger_indexes)

        # Inform selector about remaining data
        self._inform_selector_remaining_data(s, self.logs, batch, executed_triggers)

        return executed_triggers

    @pipeline_stage(PipelineType.NEW_BATCH, PipelineStage.EVALUATE_TRIGGER_POLICIES, track=True)
    def _evaluate_trigger_policies(
        self, s: ExecutionState, log: StageLog, batch: list[tuple[int, int, int]]
    ) -> list[int]:
        """Evaluate trigger policy and inform selector.

        Returns:
            List of indexes of data points that caused a trigger.
        """
        # Evaluate trigger policy
        lazy_trigger_indexes = self.trigger.inform(batch)

        # add log data
        log.info = EvaluateTriggerInfo(batch_size=len(batch), trigger_indexes=lazy_trigger_indexes)

        return lazy_trigger_indexes

    @pipeline_stage(PipelineType.NEW_BATCH, PipelineStage.EXECUTE_TRIGGERS, track=True)
    def _execute_triggers(
        self, s: ExecutionState, log: StageLog, batch: list[tuple[int, int, int]], trigger_indexes: Generator[int]
    ) -> list[int]:
        """Evaluate trigger policy, start training after trigger and inform selector.

        Args:
            s: Execution state of the pipeline executor.
            
        Returns:
            The list of the actually processed triggers
        """
        logger.info(f"Processing {len(s.triggers)} triggers in this batch.")
        s.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.EXECUTE_TRIGGERS, MsgType.GENERAL))

        previous_trigger_index = 0
        trigger_index = -1
        processed_trigger_indexes: list[int] = []
        for i, trigger_index in enumerate(trigger_indexes):
            # Run training and evaluation subpipelines
            processed_trigger_indexes.append(trigger_index)
            trigger_data = batch[previous_trigger_index : trigger_index + 1]
            previous_trigger_index = trigger_index + 1

            self._execute_single_trigger(s, self.logs, trigger_data, i, trigger_index)
            s.triggers.append(trigger_index)

            if s.maximum_triggers is not None and len(s.triggers) >= s.maximum_triggers:
                break
            
        return processed_trigger_indexes

    @pipeline_stage(PipelineType.NEW_DATA, PipelineStage.INFORM_SELECTOR_REMAINING_DATA, track=True)
    def _inform_selector_remaining_data(
        self, s: ExecutionState, log: StageLog, batch: list[tuple[int, int, int]], trigger_indexes: list[int]
    ) -> None:
        """Inform selector about remaining data."""

        # If no other trigger is coming in this batch, we  inform the Selector about the remaining data in this batch.
        if len(trigger_indexes) > 0:
            s.remaining_data = batch[trigger_indexes[-1] + 1 :]
        else:
            s.remaining_data = batch  # All data

        logger.info(f"There are {len(s.remaining_data)} data points remaining after the triggers in this batch.")

        if len(s.remaining_data) > 0:
            # These data points will be included in the next trigger
            # because we inform the Selector about them,
            # just like other batches with no trigger at all are included.
            selector_log = self.grpc.inform_selector(s.pipeline_id, s.remaining_data)
            if s.remaining_data_range is not None:
                # extend the range from last time
                s.remaining_data_range = (s.remaining_data_range[0], s.remaining_data[-1][1])
            else:
                s.remaining_data_range = (s.remaining_data[0][1], s.remaining_data[-1][1])
        else:
            selector_log = None
            s.remaining_data_range = None

        # add log data
        log.info = SelectorInformLog(
            selector_log=selector_log, remaining_data=len(s.remaining_data) > 0, trigger_indexes=trigger_indexes
        )

    # Execute trigger within batch

    @pipeline_stage(PipelineType.TRIGGER, PipelineStage.EXECUTE_SINGLE_TRIGGER, track=True)
    def _execute_single_trigger(
        self,
        s: ExecutionState,
        log: StageLog,
        trigger_data: list[tuple[int, int, int]],
        trigger_i: int,
        trigger_index: int,
    ) -> None:
        """Execute trigger within batch.

        Args:
            s: Execution state of the pipeline executor.
            trigger_data: Data points used for the training caused by the trigger.
            trigger_i: Index of the trigger in the batch.
            trigger_index: Index of the trigger in the data.
        """
        s.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.EXECUTE_SINGLE_TRIGGER, MsgType.GENERAL))

        # trigger_id: identifier of the trigger received from the selector
        trigger_id, num_samples_in_trigger = self._inform_selector_about_trigger(
            s, self.logs, trigger_data, trigger_i, trigger_index
        )

        if num_samples_in_trigger > 0:
            first_timestamp, last_timestamp = PipelineExecutor._get_trigger_timespan(s, trigger_i == 0, trigger_data)
            s.remaining_data_range = None
            training_id, model_id = self._train_and_store_model(s, self.logs, trigger_id)

            if s.pipeline_config.evaluation:
                self._evaluate_and_store_results(s, self.logs, trigger_id, training_id, model_id)

        else:
            first_timestamp, last_timestamp = None, None
            logger.info(f"Skipping training on empty trigger {trigger_index}]")

        log.info = TriggerExecutionInfo(
            trigger_i=trigger_i,
            trigger_index=trigger_index,
            trigger_id=trigger_id,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
        )

    @pipeline_stage(PipelineType.TRIGGER, PipelineStage.INFORM_SELECTOR_AND_TRIGGER, track=True)
    def _inform_selector_about_trigger(
        self,
        s: ExecutionState,
        log: StageLog,
        trigger_data: list[tuple[int, int, int]],
        trigger_i: int,
        trigger_index: int,
    ) -> tuple[int, int]:
        """Inform selector about data until trigger and notify about trigger.

        Returns:
            trigger id from selector and number of samples in trigger.
        """
        s.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.INFORM_SELECTOR_AND_TRIGGER, MsgType.GENERAL))

        # This call informs the selector about the data until (and including) the data point that caused the trigger
        # and then also notifies it about the triggering. This means the next training call on trigger_id will
        # guarantee that all data until that point has been processed by the selector.
        trigger_id, selector_log = self.grpc.inform_selector_and_trigger(s.pipeline_id, trigger_data)
        num_samples_in_trigger = self.grpc.get_number_of_samples(s.pipeline_id, trigger_id)

        # add log data
        log.info = SelectorInformTriggerInfo(
            trigger_i=trigger_i,
            trigger_index=trigger_index,
            trigger_id=trigger_id,
            selector_log=selector_log,
            num_samples_in_trigger=num_samples_in_trigger,
        )
        return trigger_id, num_samples_in_trigger

    # Training

    @pipeline_stage(PipelineType.TRAINING, PipelineStage.TRAIN_AND_STORE_MODEL, track=True)
    def _train_and_store_model(self, s: ExecutionState, log: StageLog, trigger_id: int) -> tuple[int, int]:
        """Train a new model on batch data and store it."""

        training_id = self._train(s, self.logs, trigger_id)
        self._training_completed(s, self.logs, trigger_id)
        model_id = self._store_trained_model(s, self.logs, trigger_id, training_id)

        s.trained_models.append(model_id)

        s.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.EXECUTE_TRIGGERS, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )
        return training_id, model_id

    @pipeline_stage(PipelineType.TRAINING, PipelineStage.TRAIN, track=True)
    def _train(self, s: ExecutionState, log: StageLog, trigger_id: int) -> int:
        """Run training for trigger on GPU and block until done."""

        logger.info(f"Running training for trigger {trigger_id}")
        s.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.TRAIN, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )
        num_samples_to_pass_per_trigger = s.pipeline_config.training.num_samples_to_pass or []
        current_trigger_index = len(s.triggers)
        if current_trigger_index <= len(num_samples_to_pass_per_trigger) - 1:
            num_samples_to_pass = num_samples_to_pass_per_trigger[current_trigger_index]
        else:
            num_samples_to_pass = None

        s.current_training_id = self.grpc.start_training(
            s.pipeline_id,
            trigger_id,
            s.pipeline_config.model_dump(by_alias=True),
            s.previous_model_id,
            num_samples_to_pass
        )
        trainer_log = self.grpc.wait_for_training_completion(s.current_training_id, s.pipeline_id, trigger_id)

        # add log data
        log.info = TrainingInfo(
            trigger_id=trigger_id,
            training_id=s.current_training_id,
            trainer_log=trainer_log,
        )

        return s.current_training_id

    @pipeline_stage(PipelineType.TRAINING, PipelineStage.TRAINING_COMPLETED, track=False)
    def _training_completed(self, s: ExecutionState, log: StageLog, training_id: int) -> None:
        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.TRAINING_COMPLETED, MsgType.ID, id_submsg(IdType.TRAINING, training_id), True
            )
        )
        logger.info(f"Training {training_id} completed")

    @pipeline_stage(PipelineType.TRAINING, PipelineStage.STORE_TRAINED_MODEL, track=True)
    def _store_trained_model(self, s: ExecutionState, log: StageLog, trigger_id: int, training_id: int) -> int:
        """Stores a trained model and returns the model id."""
        s.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.STORE_TRAINED_MODEL, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )

        model_id = self.grpc.store_trained_model(training_id)

        # Only if the pipeline actually wants to continue the training on it, we set previous model.
        if s.pipeline_config.training.use_previous_model:
            s.previous_model_id = model_id

        # add log data
        log.info = StoreModelInfo(trigger_id=trigger_id, training_id=training_id, id_model=model_id)

        return model_id

    # Evaluation

    @pipeline_stage(PipelineType.EVALUATION, PipelineStage.EVALUATE, track=True)
    def _evaluate_and_store_results(
        self, s: ExecutionState, log: StageLog, trigger_id: int, training_id: int, model_id: int
    ) -> None:
        """Evaluate the trained model and store the results."""
        evaluations = self._evaluate(s, self.logs, trigger_id, training_id, model_id)
        self._evaluation_completed(s, self.logs)
        self._store_evaluation_results(s, self.logs, trigger_id, evaluations)

    @pipeline_stage(PipelineType.EVALUATION, PipelineStage.EVALUATE, track=True)
    def _evaluate(
        self, s: ExecutionState, log: StageLog, trigger_id: int, training_id: int, model_id: int
    ) -> dict[int, EvaluationStatusReporter]:
        s.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.EVALUATE, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )

        evaluations = self.grpc.start_evaluation(model_id, s.pipeline_config)
        self.grpc.wait_for_evaluation_completion(training_id, evaluations)

        # add log data
        log.info = EvaluationInfo(trigger_id=trigger_id, training_id=training_id, id_model=model_id)

        return evaluations

    @pipeline_stage(PipelineType.EVALUATION, PipelineStage.EVALUATION_COMPLETED)
    def _evaluation_completed(self, s: ExecutionState, log: StageLog) -> None:
        pass  # nothing to do

    @pipeline_stage(PipelineType.EVALUATION, PipelineStage.STORE_EVALUATION_RESULTS, track=True)
    def _store_evaluation_results(
        self, s: ExecutionState, log: StageLog, trigger_id: int, evaluations: dict[int, EvaluationStatusReporter]
    ) -> None:
        assert s.pipeline_config.evaluation
        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.STORE_EVALUATION_RESULTS,
                MsgType.ID,
                id_submsg(IdType.TRIGGER, trigger_id),
            )
        )

        writer_names: set[str] = set(s.pipeline_config.evaluation.result_writers)
        writers = [self._init_evaluation_writer(s, name, trigger_id) for name in writer_names]
        self.grpc.store_evaluation_results(writers, evaluations)

    # Teardown

    @pipeline_stage(PipelineType.MAIN, PipelineStage.DONE)
    def _done(self, s: ExecutionState, log: StageLog) -> None:
        s.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.DONE, MsgType.GENERAL))
        self.logs.materialize(s.log_directory, mode="final")

    @pipeline_stage(PipelineType.MAIN, PipelineStage.EXIT)
    def _exit(self, s: ExecutionState, log: StageLog) -> None:
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

    @staticmethod
    def _get_trigger_timespan(
        s: ExecutionState, is_first_trigger_data: bool, trigger_data: list[tuple[int, int, int]]
    ) -> tuple[int, int]:
        if is_first_trigger_data:
            # now it is the first trigger in this batch. Triggering_data can be empty.
            # when it is indeed empty, then there is remaining data in the last batch
            # because num_samples_in_trigger is not 0.
            assert len(trigger_data) > 0 or s.remaining_data_range is not None

            if s.remaining_data_range is not None:
                first_timestamp = s.remaining_data_range[0]
                last_timestamp = s.remaining_data_range[1] if len(trigger_data) == 0 else trigger_data[-1][1]
            else:
                first_timestamp = trigger_data[0][1]
                last_timestamp = trigger_data[-1][1]
        else:
            assert len(trigger_data) > 0
            # since num_samples_in_trigger is not 0, we are sure that trigger_data is not empty
            first_timestamp = trigger_data[0][1]
            last_timestamp = trigger_data[-1][1]

        return first_timestamp, last_timestamp

    def _init_evaluation_writer(self, s: ExecutionState, name: str, trigger_id: int) -> LogResultWriter:
        return s.supervisor_supported_eval_result_writers[name](s.pipeline_id, trigger_id, self.state.eval_directory)

    def _shutdown_trainer(self) -> None:
        if self.state.current_training_id is not None:
            self.grpc.stop_training_at_trainer_server(self.state.current_training_id)


def execute_pipeline(options: PipelineOptions) -> None:
    try:
        PipelineExecutor(options).run()

    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        options.exception_queue.put(exception_msg)
        sys.exit(EXCEPTION_EXITCODE)
