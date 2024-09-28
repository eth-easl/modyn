# pylint: disable=unused-argument
from __future__ import annotations

import logging
import sys
import traceback
import types
from collections.abc import Callable, Generator
from datetime import datetime, timedelta
from time import sleep
from typing import Concatenate, TypeVar, cast

import pandas as pd
from typing_extensions import ParamSpec

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
from modyn.supervisor.internal.pipeline_executor.evaluation_executor import (
    EvaluationExecutor,
)
from modyn.supervisor.internal.pipeline_executor.models import (
    ConfigLogs,
    EvaluateTriggerInfo,
    ExecutionState,
    FetchDataInfo,
    PipelineExecutionParams,
    PipelineLogs,
    ProcessNewDataInfo,
    SelectorInformInfo,
    SelectorInformTriggerInfo,
    StageLog,
    StoreModelInfo,
    TrainingInfo,
    TriggerExecutionInfo,
)
from modyn.supervisor.internal.triggers import Trigger
from modyn.supervisor.internal.triggers.trigger import TriggerContext
from modyn.supervisor.internal.triggers.utils.factory import instantiate_trigger
from modyn.supervisor.internal.utils import TrainingStatusReporter
from modyn.utils.timer import timed_generator
from modyn.utils.utils import current_time_micros

logger = logging.getLogger(__name__)
EXCEPTION_EXITCODE = 8


P = ParamSpec("P")  # parameters of pipeline stage
R = TypeVar("R")  # result of pipeline stage

G = TypeVar("G")  # generator type

_pipeline_stage_parents: dict[str, tuple[int, list[str]]] = {PipelineStage.MAIN.name: (-1, [])}
"""Automatically filled parent relationships for pipeline stages."""


def pipeline_stage(  # type: ignore[no-untyped-def]
    stage: PipelineStage,
    parent: PipelineStage | list[PipelineStage] | None = None,
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
    _pipeline_stage_parents[stage.name] = (
        -1,  # sequential execution index; -1 as not yet determined
        (
            [parent.name]
            if parent is not None and not isinstance(parent, list)
            else ([p.name for p in parent] if isinstance(parent, list) else [])
        ),
    )

    def wrapper_outer(  # type: ignore[no-untyped-def]
        func: Callable[Concatenate[PipelineExecutor, ExecutionState, StageLog, P], R],
    ):
        def wrapper(
            self: PipelineExecutor,
            state: ExecutionState,
            logs: PipelineLogs,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            """Measures the time for each stage and logs the pipeline state."""

            def patch_generator_timer(gen: R, stage_log: StageLog) -> R:  # type: ignore[misc]
                """As generators will return immediate we have to add time for
                each yield after the decorator returned.

                For doing so we wrap the generator with this function.
                """
                try:
                    for item, time_ in timed_generator(gen):  # type: ignore
                        stage_log.duration = (stage_log.duration or timedelta(0)) + timedelta(milliseconds=time_)
                        yield item
                finally:
                    report_results(stage_log)

            def report_results(stage_log: StageLog) -> None:
                """For generators we should only report logs and tracking info
                once the generator is finalized.

                I.e. fully iterated or garbage collected. In the non-
                generator case we can report immediately.
                """
                # if stage reported additional logs, we make the log available to the pipeline in a dataframe
                if track and stage_log.info:
                    # ensure df exists
                    old_df = state.tracking.get(stage_log.id, None)
                    columns = old_df.columns if old_df is not None else stage_log.df_columns(extended=True)
                    if (new_row := stage_log.df_row(extended=True)) is not None:
                        new_df = pd.DataFrame([new_row], columns=columns)
                        state.tracking[stage_log.id] = pd.concat([old_df, new_df]) if old_df is not None else new_df

                # record logs
                if log:
                    logs.supervisor_logs.stage_runs.append(stage_log)
                    logger.info(f"[pipeline {state.pipeline_id}] Finished <{stage.name}>.")

            state.stage = stage
            if stage not in state.seen_pipeline_stages:
                _pipeline_stage_parents[stage.name] = (
                    len(state.seen_pipeline_stages),
                    _pipeline_stage_parents[stage.name][1],
                )
                state.seen_pipeline_stages.add(stage)

            if log:
                logger.info(f"[pipeline {state.pipeline_id}] Entering <{stage}>.")

            # execute stage
            stage_seq_num = state.stage_id_seq_counters.get(stage.name, 0)
            state.stage_id_seq_counters[stage.name] = stage_seq_num + 1
            epoch_micros_start = current_time_micros()
            stage_log = StageLog(
                id=stage.name,
                id_seq_num=stage_seq_num,
                start=datetime.now(),
                batch_idx=state.current_batch_index,
                sample_idx=state.current_sample_index,
                sample_time=state.current_sample_time,
                trigger_idx=len(state.triggers),
            )
            result = func(self, state, stage_log, *args, **kwargs)  # type: ignore[call-arg]
            stage_log.end = datetime.now()
            stage_log.duration = timedelta(microseconds=current_time_micros() - epoch_micros_start)
            state.stage = stage  # restore stage as child pipeline might have changed it

            if isinstance(result, types.GeneratorType):
                # If the result is a generator, wrap it with the timed_generator so that every yield will be timed
                # and added to the log time. The adding happens post-return when the generator is actually iterated.
                # The wrapped generator adjusts the duration of the stage log and tracking dataframe.
                result = cast(R, patch_generator_timer(result, stage_log))  # type: ignore[arg-type]

            else:
                report_results(stage_log)

            # result of stage function
            return result

        return wrapper

    return wrapper_outer


class PipelineExecutor:
    def __init__(self, options: PipelineExecutionParams) -> None:
        self.stage = PipelineStage.INIT
        self.state = ExecutionState(**vars(options))
        self.logs = PipelineLogs(
            pipeline_id=options.pipeline_id,
            pipeline_stages=_pipeline_stage_parents,
            config=ConfigLogs(system=options.modyn_config, pipeline=options.pipeline_config),
            experiment=options.experiment_mode,
            start_replay_at=options.start_replay_at,
            stop_replay_at=options.stop_replay_at,
        )
        """Execution state of the pipeline executor."""

        # pipeline controllers objects
        self.trigger = self._setup_trigger()
        self.grpc = GRPCHandler(self.state.modyn_config.model_dump(by_alias=True))
        self.eval_executor = EvaluationExecutor(
            options.pipeline_id,
            options.pipeline_logdir,
            options.modyn_config,
            options.pipeline_config,
            self.grpc,
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

        self._post_pipeline_evaluation_checkpoint(self.state, self.logs)
        self._post_pipeline_evaluation(self.state, self.logs)

        self._exit(self.state, self.logs)

        logger.info(f"[pipeline {self.state.pipeline_id}] Execution done. Persist log.")

    # ------------------------------------------------ Pipeline stages ----------------------------------------------- #

    # Setup

    @pipeline_stage(PipelineStage.INIT, parent=PipelineStage.MAIN, log=False)
    def _init(self, s: ExecutionState, log: StageLog) -> None:
        s.max_timestamp = s.start_timestamp
        self.logs.materialize(s.log_directory, mode="initial")
        if s.pipeline_config.training.initial_model == "pretrained":
            s.previous_model_id = s.pipeline_config.training.initial_model_id

    @pipeline_stage(PipelineStage.INIT_CLUSTER_CONNECTION, parent=PipelineStage.MAIN)
    def _init_cluster_connection(self, s: ExecutionState, log: StageLog) -> None:
        s.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.INIT_CLUSTER_CONNECTION, MsgType.GENERAL))
        self.grpc.init_cluster_connection()

    # Replay Data (experiment mode)

    @pipeline_stage(PipelineStage.REPLAY_DATA, parent=PipelineStage.MAIN)
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
            # setting sample info here to have correct logs in process_new_data
            s.current_sample_time = replay_data[0][1] if replay_data else s.start_timestamp
            s.current_sample_index = replay_data[0][0] if replay_data else 0

            self._process_new_data(s, self.logs, replay_data, request_time)

            # to identify the dataset end in the logs
            s.current_sample_time = replay_data[-1][1] if replay_data else s.start_timestamp
            s.current_sample_index = replay_data[-1][0] if replay_data else 0

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

    @pipeline_stage(PipelineStage.SERVE_ONLINE_DATA, parent=PipelineStage.MAIN)
    def _serve_online_data(self, s: ExecutionState, log: StageLog) -> None:
        """Run pipeline in production mode fetching new data until pipeline is
        stopped."""
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

    @pipeline_stage(PipelineStage.FETCH_NEW_DATA, parent=PipelineStage.SERVE_ONLINE_DATA, track=True)
    def _fetch_new_data(self, s: ExecutionState, log: StageLog) -> int:
        """Try to fetch new data from the dataset and process it.

        Returns:
            The number of triggers that occurred during the processing of the new data.
        """
        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.FETCH_NEW_DATA,
                MsgType.DATASET,
                dataset_submsg(s.dataset_id),
            )
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
                max(timestamp for (_, timestamp, _) in fetched_data) if len(fetched_data) > 0 else s.max_timestamp
            )
            largest_keys.update({key for (key, timestamp, _) in fetched_data if timestamp == s.max_timestamp})

            # setting sample info here to have correct logs in process_new_data
            s.current_sample_time = fetched_data[0][1] if fetched_data else s.start_timestamp
            s.current_sample_index = fetched_data[0][0] if fetched_data else 0

            # process new data and invoke triggers
            trigger_indexes = trigger_indexes + self._process_new_data(s, self.logs, fetched_data, fetch_time)
            num_samples += len(fetched_data)

        s.previous_largest_keys = largest_keys

        # log extra information
        log.info = FetchDataInfo(num_samples=num_samples, trigger_indexes=trigger_indexes)

        return len(trigger_indexes)

    @pipeline_stage(PipelineStage.WAIT_FOR_NEW_DATA, parent=PipelineStage.SERVE_ONLINE_DATA)
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

    @pipeline_stage(
        PipelineStage.PROCESS_NEW_DATA,
        parent=[PipelineStage.REPLAY_DATA, PipelineStage.FETCH_NEW_DATA],
        track=True,
    )
    def _process_new_data(
        self,
        s: ExecutionState,
        log: StageLog,
        new_data: list[tuple[int, int, int]],
        fetch_time: int,
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
                counter_submsg(
                    CounterAction.CREATE,
                    {"title": "Processing New Samples", "new_data_len": new_data_len},
                ),
            )
        )

        for i in range(0, new_data_len, s.selector_batch_size):
            batch = new_data[i : i + s.selector_batch_size]
            batch_size = s.selector_batch_size if i + s.selector_batch_size < new_data_len else new_data_len - i
            if batch_size > 0:
                s.current_sample_time = batch[0][1]  # update sample time

            s.pipeline_status_queue.put(
                pipeline_stage_msg(
                    PipelineStage.PROCESS_NEW_DATA,
                    MsgType.COUNTER,
                    counter_submsg(CounterAction.UPDATE, {"increment": batch_size}),
                )
            )

            trigger_indexes += self._process_new_data_batch(s, self.logs, batch)
            s.current_batch_index += 1

            if s.maximum_triggers is not None and len(s.triggers) >= s.maximum_triggers:
                logger.info(f"Reached trigger limit ({s.maximum_triggers}), exiting.")
                break

        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.NEW_DATA_HANDLED,
                MsgType.COUNTER,
                counter_submsg(CounterAction.CLOSE),
            )
        )

        # log extra information
        log.info = ProcessNewDataInfo(
            fetch_time=fetch_time,
            num_samples=new_data_len,
            trigger_indexes=trigger_indexes,
        )
        self.logs.materialize(s.log_directory, mode="increment")

        return trigger_indexes

    # Process new data BATCH

    @pipeline_stage(
        PipelineStage.PROCESS_NEW_DATA_BATCH,
        parent=PipelineStage.PROCESS_NEW_DATA,
        track=True,
    )
    def _process_new_data_batch(self, s: ExecutionState, log: StageLog, batch: list[tuple[int, int, int]]) -> list[int]:
        """Process new data in batches and evaluate trigger policies in
        batches.

        Args:
            s: Execution state of the pipeline executor.
            log: Log of the current stage.
            batch: List of tuples (key, timestamp, label) of new data points.

        Returns:
            List of indexes of data points that caused a trigger.
        """

        # Evaluate trigger policy and inform selector
        lazy_trigger_indexes = self._evaluate_trigger_policy(s, self.logs, batch)

        # Handle triggers within batch (training & evaluation subpipelines)
        handled_triggers = self._handle_triggers(s, self.logs, batch, lazy_trigger_indexes)

        # Inform selector about remaining data
        self._inform_selector_remaining_data(s, self.logs, batch, handled_triggers)

        return handled_triggers

    @pipeline_stage(
        PipelineStage.EVALUATE_TRIGGER_POLICY,
        parent=PipelineStage.PROCESS_NEW_DATA_BATCH,
        track=True,
    )
    def _evaluate_trigger_policy(
        self, s: ExecutionState, log: StageLog, batch: list[tuple[int, int, int]]
    ) -> Generator[int, None, None]:
        """Evaluate trigger policy and inform selector.

        Returns:
            List of indexes of data points that caused a trigger.
        """
        try:
            # add log data
            log.info = EvaluateTriggerInfo(batch_size=len(batch))

            # Evaluate trigger policy
            for trigger_index, t_micros in timed_generator(self.trigger.inform(batch, log.info.trigger_evaluation_log)):
                log.info.trigger_indexes.append(trigger_index)
                log.info.trigger_eval_times.append(int(t_micros))
                yield trigger_index

        finally:
            assert log.info
            logger.info(f"There were {len(log.info.trigger_indexes)} triggers in this batch.")

    @pipeline_stage(
        PipelineStage.HANDLE_TRIGGERS,
        parent=PipelineStage.PROCESS_NEW_DATA_BATCH,
        track=True,
    )
    def _handle_triggers(
        self,
        s: ExecutionState,
        log: StageLog,
        batch: list[tuple[int, int, int]],
        lazy_trigger_indexes: Generator[int, None, None],
    ) -> list[int]:
        """Evaluate trigger policy, start training after trigger and inform
        selector.

        Args:
            s: Execution state of the pipeline executor.

        Returns:
            The list of the actually processed triggers
        """
        s.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.HANDLE_TRIGGERS, MsgType.GENERAL))

        previous_trigger_index = 0
        trigger_index = -1
        trigger_indexes: list[int] = []
        for i, trigger_index in enumerate(lazy_trigger_indexes):
            # Run training and evaluation substages
            trigger_indexes.append(trigger_index)
            trigger_data = batch[previous_trigger_index : trigger_index + 1]
            previous_trigger_index = trigger_index + 1

            if len(trigger_data) > 0:
                s.current_sample_time = trigger_data[0][1]  # update sample time

            self._handle_single_trigger(s, self.logs, trigger_data, i, trigger_index)
            s.triggers.append(trigger_index)
            s.current_sample_index += len(trigger_data)

            self.logs.materialize(s.log_directory, mode="increment")  # materialize after every trigger

            if s.maximum_triggers is not None and len(s.triggers) >= s.maximum_triggers:
                break

        return trigger_indexes

    @pipeline_stage(
        PipelineStage.INFORM_SELECTOR_REMAINING_DATA,
        parent=PipelineStage.PROCESS_NEW_DATA_BATCH,
        track=True,
    )
    def _inform_selector_remaining_data(
        self,
        s: ExecutionState,
        log: StageLog,
        batch: list[tuple[int, int, int]],
        trigger_indexes: list[int],
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
                s.remaining_data_range = (
                    s.remaining_data_range[0],
                    s.remaining_data[-1][1],
                )
            else:
                s.remaining_data_range = (
                    s.remaining_data[0][1],
                    s.remaining_data[-1][1],
                )
        else:
            selector_log = None
            s.remaining_data_range = None

        # add log data
        log.info = SelectorInformInfo(
            selector_log=selector_log,
            remaining_data=len(s.remaining_data) > 0,
            trigger_indexes=trigger_indexes,
        )

    # Handle trigger within batch

    @pipeline_stage(
        PipelineStage.HANDLE_SINGLE_TRIGGER,
        parent=PipelineStage.HANDLE_TRIGGERS,
        track=True,
    )
    def _handle_single_trigger(
        self,
        s: ExecutionState,
        log: StageLog,
        trigger_data: list[tuple[int, int, int]],
        trigger_i: int,
        trigger_index: int,
    ) -> None:
        """Handle trigger within batch.

        Args:
            s: Execution state of the pipeline executor.
            trigger_data: Data points used for the training caused by the trigger.
            trigger_i: Index of the trigger in the batch.
            trigger_index: Index of the trigger in the data.
        """
        s.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.HANDLE_SINGLE_TRIGGER, MsgType.GENERAL))

        # trigger_id: identifier of the trigger received from the selector
        trigger_id, num_samples_in_trigger = self._inform_selector_about_trigger(
            s, self.logs, trigger_data, trigger_i, trigger_index
        )

        if num_samples_in_trigger > 0:
            first_timestamp, last_timestamp = PipelineExecutor._get_trigger_timespan(s, trigger_i == 0, trigger_data)
            s.remaining_data_range = None
            training_id, model_id = self._train_and_store_model(s, self.logs, trigger_id)

            # fetch latest training time from tracker data
            tracking_df_train = s.tracking[PipelineStage.TRAIN.name]
            max_trigger_idx = tracking_df_train['trigger_idx'].max()
            time_at_trainer = float(tracking_df_train[tracking_df_train["trigger_idx"] == max_trigger_idx]["train_time_at_trainer"][0])
            last_training_seconds = time_at_trainer / 1_000  # ms to s
            self.trigger.inform_new_model(model_id, num_samples_in_trigger, last_training_seconds)

            if s.pipeline_config.evaluation:
                self._evaluate_and_store_results(
                    s,
                    self.logs,
                    trigger_id,
                    training_id,
                    model_id,
                    first_timestamp,
                    last_timestamp,
                )

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

    @pipeline_stage(
        PipelineStage.INFORM_SELECTOR_ABOUT_TRIGGER,
        parent=PipelineStage.HANDLE_SINGLE_TRIGGER,
        track=True,
    )
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
        s.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.INFORM_SELECTOR_ABOUT_TRIGGER, MsgType.GENERAL))

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

    @pipeline_stage(
        PipelineStage.TRAIN_AND_STORE_MODEL,
        parent=PipelineStage.HANDLE_SINGLE_TRIGGER,
        track=True,
    )
    def _train_and_store_model(self, s: ExecutionState, log: StageLog, trigger_id: int) -> tuple[int, int]:
        """Train a new model on batch data and store it."""

        training_id = self._train(s, self.logs, trigger_id)
        self._training_completed(s, self.logs, trigger_id)
        model_id = self._store_trained_model(s, self.logs, trigger_id, training_id)

        s.trained_models.append(model_id)

        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.HANDLE_TRIGGERS,
                MsgType.ID,
                id_submsg(IdType.TRIGGER, trigger_id),
            )
        )
        return training_id, model_id

    @pipeline_stage(PipelineStage.TRAIN, parent=PipelineStage.TRAIN_AND_STORE_MODEL, track=True)
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
            s.pipeline_config.training,
            s.pipeline_config.data,
            s.previous_model_id,
            num_samples_to_pass,
        )

        total_samples = self.grpc.get_number_of_samples(s.pipeline_id, trigger_id)
        status_bar_scale = self.grpc.get_status_bar_scale(s.pipeline_id)
        training_reporter = TrainingStatusReporter(
            self.state.training_status_queue,
            trigger_id,
            s.current_training_id,
            total_samples,
            status_bar_scale,
        )

        trainer_log = self.grpc.wait_for_training_completion(s.current_training_id, training_reporter)

        # add log data
        log.info = TrainingInfo(
            trigger_id=trigger_id,
            training_id=s.current_training_id,
            trainer_log=trainer_log,
        )

        return s.current_training_id

    @pipeline_stage(
        PipelineStage.TRAINING_COMPLETED,
        parent=PipelineStage.TRAIN_AND_STORE_MODEL,
        track=False,
    )
    def _training_completed(self, s: ExecutionState, log: StageLog, training_id: int) -> None:
        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.TRAINING_COMPLETED,
                MsgType.ID,
                id_submsg(IdType.TRAINING, training_id),
                True,
            )
        )
        logger.info(f"Training {training_id} completed")

    @pipeline_stage(
        PipelineStage.STORE_TRAINED_MODEL,
        parent=PipelineStage.TRAIN_AND_STORE_MODEL,
        track=True,
    )
    def _store_trained_model(self, s: ExecutionState, log: StageLog, trigger_id: int, training_id: int) -> int:
        """Stores a trained model and returns the model id."""
        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.STORE_TRAINED_MODEL,
                MsgType.ID,
                id_submsg(IdType.TRIGGER, trigger_id),
            )
        )

        model_id = self.grpc.store_trained_model(training_id)

        # Only if the pipeline actually wants to continue the training on it, we set previous model.
        if s.pipeline_config.training.use_previous_model:
            s.previous_model_id = model_id

        # add log data
        log.info = StoreModelInfo(trigger_id=trigger_id, training_id=training_id, id_model=model_id)

        return model_id

    # Evaluation

    @pipeline_stage(PipelineStage.EVALUATE, parent=PipelineStage.HANDLE_SINGLE_TRIGGER, track=True)
    def _evaluate_and_store_results(
        self,
        s: ExecutionState,
        log: StageLog,
        trigger_id: int,
        training_id: int,
        model_id: int,
        first_timestamp: int,
        last_timestamp: int,
    ) -> None:
        """Evaluate the trained model and store the results."""
        s.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.EVALUATE,
                MsgType.ID,
                id_submsg(IdType.TRIGGER, trigger_id),
            )
        )
        logs = self.eval_executor.run_pipeline_evaluations(
            log,
            trigger_id,
            training_id,
            model_id,
            first_timestamp,
            last_timestamp,
        )
        self.logs.supervisor_logs.merge(logs.stage_runs)

    # Teardown

    @pipeline_stage(PipelineStage.DONE, parent=PipelineStage.MAIN)
    def _done(self, s: ExecutionState, log: StageLog) -> None:
        s.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.DONE, MsgType.GENERAL))
        self.logs.pipeline_stages = _pipeline_stage_parents  # now includes chronology info

    @pipeline_stage(
        PipelineStage.POST_EVALUATION_CHECKPOINT,
        parent=PipelineStage.MAIN,
        log=False,
        track=False,
    )
    def _post_pipeline_evaluation_checkpoint(self, s: ExecutionState, log: StageLog) -> None:
        """Stores evaluation relevant information so that the evaluator can be
        started on this pipeline run again."""

        if not s.pipeline_config.evaluation:
            return

        self.logs.materialize(s.log_directory, mode="increment")
        self.eval_executor.register_tracking_info(
            tracking_dfs=s.tracking, dataset_end_time=self.state.current_sample_time
        )
        self.eval_executor.create_snapshot()

    @pipeline_stage(PipelineStage.POST_EVALUATION, parent=PipelineStage.MAIN, log=False, track=False)
    def _post_pipeline_evaluation(self, s: ExecutionState, log: StageLog) -> None:
        """Evaluate the trained model and store the results."""
        if not s.pipeline_config.evaluation:
            return

        eval_logs = self.eval_executor.run_post_pipeline_evaluations()
        self.logs.supervisor_logs.merge(eval_logs.stage_runs)

    @pipeline_stage(PipelineStage.EXIT, parent=PipelineStage.MAIN)
    def _exit(self, s: ExecutionState, log: StageLog) -> None:
        self.logs.materialize(s.log_directory, mode="final")

    # ---------------------------------------------------- Helpers --------------------------------------------------- #

    # setup

    def _setup_trigger(self) -> Trigger:
        trigger = instantiate_trigger(self.state.pipeline_config.trigger.id, self.state.pipeline_config.trigger)
        trigger.init_trigger(
            TriggerContext(
                pipeline_id=self.state.pipeline_id,
                pipeline_config=self.state.pipeline_config,
                modyn_config=self.state.modyn_config,
                base_dir=self.state.eval_directory,
            )
        )
        if self.state.previous_model_id is not None:
            trigger.inform_new_model(self.state.previous_model_id)

        return trigger

    # pipeline run

    @staticmethod
    def _get_trigger_timespan(
        s: ExecutionState,
        is_first_trigger_data: bool,
        trigger_data: list[tuple[int, int, int]],
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

    def _shutdown_trainer(self) -> None:
        if self.state.current_training_id is not None:
            self.grpc.stop_training_at_trainer_server(self.state.current_training_id)


def execute_pipeline(options: PipelineExecutionParams) -> None:
    try:
        PipelineExecutor(options).run()

    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        options.exception_queue.put(exception_msg)
        sys.exit(EXCEPTION_EXITCODE)
