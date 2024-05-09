from __future__ import annotations

# TODO (THIS PR)!!! check history and reimplement changes (take ours merge happened here)
import datetime
import logging
import sys
import traceback
from time import sleep
from typing import Any, Callable

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
    ExecutionState,
    NewDataRequestLog,
    PipelineLogs,
    PipelineOptions,
    RegisteredStage,
    SelectorInformLog,
    StageRunLog,
    TriggerBatchTimeLog,
    TriggerLog,
)

logger = logging.getLogger(__name__)
EXCEPTION_EXITCODE = 8

PipelineRegistry = dict[PipelineStage, RegisteredStage]

# Dynamically registered pipeline stages (via decorators)
# for subpipeline hierarchy, see `PIPELINE.md`
main_pipeline: PipelineRegistry  = {}
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
    next: PipelineStage | None = None, 
    logging: bool = True
) -> Callable:
    """Decorator to register a pipeline stage handler function."""

    def wrapper_outer(func: Callable[[Any,], PipelineStage | None]) -> Callable[[Any,], PipelineStage | None]:
        assert stage not in pipeline
        pipeline[stage] = PipelineStage(stage=stage, func=func, next=next, logging=logging)
        def wrapper(*args: Any, **kwargs: Any) -> PipelineStage | None:
            return func(*args, **kwargs)
        return wrapper
    return wrapper_outer

    
    # TODO: pipeline artifacts

class PipelineExecutor:
    def __init__(self, options: PipelineOptions) -> None:
        self.stage = PipelineStage.INIT
        self.state = ExecutionState(**dict(options))
        self.logs = PipelineLogs(options.pipeline_id, options.modyn_config, options.pipeline_config)
        """Execution state of the pipeline executor."""
        
        # pipeline controllers objects
        self.trigger = self._setup_trigger()
        self.grpc = GRPCHandler(
            self.state.modyn_config, self.state.pipeline_status_queue, self.state.training_status_queue, self.state.eval_status_queue
        )

    # -------------------------------------------- Pipeline orchestration -------------------------------------------- #
        
    def execute(
        self, 
        pipeline: dict[PipelineStage, RegisteredStage], 
        initial_stage: PipelineStage = PipelineStage.INIT
    ) -> None:
        """Coordinates all pipelines stages until the pipeline execution is finished.
        
        Measures the time for each stage and logs the pipeline state.
        """
        self.stage = initial_stage
        while True:
            stage = pipeline[self.stage]
            
            if stage.logging:
                logger.info(f"[pipeline {self.state.pipeline_id}] Entering <{stage.stage}>.")
            
            # execute stage
            start = datetime.datetime.now()
            returned_stage = stage.func(self)
            end = datetime.datetime.now()
            
            # record logs
            if stage.logging:
                self.logs.supervisor.stage_runs.append(StageRunLog(id=stage.stage, start=start, end=end))
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
    
    @register_stage(main_pipeline, PipelineStage.INIT, next=PipelineStage.INIT_CLUSTER_CONNECTION, logging=False)
    def _init(self) -> None:
        self.logs.materialize(self.state.log_directory, mode="initial")
        if self.state.pipeline_config.training.initial_model == "pretrained":
            self.state.previous_model_id = self.state.pipeline_config.training.initial_model_id

    @register_stage(main_pipeline, PipelineStage.INIT_CLUSTER_CONNECTION, next=PipelineStage._FORK_DATA_STRATEGY)
    def _init_cluster_connection(self) -> None:
        self.state.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.INIT_CLUSTER_CONNECTION, MsgType.GENERAL))
        self.grpc.init_cluster_connection()

    @register_stage(main_pipeline, PipelineStage._FORK_DATA_STRATEGY, next=PipelineStage.DONE, logging=False)
    def _fork_data_strategy(self) -> None:
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

    # Replay Data
    
    @register_stage(replay_data_pipeline, PipelineStage.REPLAY_DATA, next=PipelineStage.REPLAY_DATA_DONE)
    def replay_data(self) -> None:
        assert self.state.start_replay_at is not None, "Cannot call replay_data when start_replay_at is None"
        dataset_id = self.state.pipeline_config["data"]["dataset_id"]
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.REPLAY_DATA, MsgType.DATASET, dataset_submsg(dataset_id))
        )
        logger.info("Starting data replay.")

        if self.state.stop_replay_at is None:
            replay_data_generator = self.grpc.get_new_data_since(dataset_id, self.state.start_replay_at)
        else:
            replay_data_generator = self.grpc.get_data_in_interval(dataset_id, self.state.start_replay_at, self.state.stop_replay_at)

        for replay_data, request_time in replay_data_generator:
            assert isinstance(replay_data, list)
            assert isinstance(request_time, int)
            self.logs.supervisor.new_data_requests.append(NewDataRequestLog(time=request_time, num_items=len(replay_data)))
            
            # Run new data subpipeline
            self.state.new_data = replay_data
            self.execute(new_data_pipeline, initial_stage=PipelineStage.EVALUATE_TRIGGER_ON_BATCH)
            
            self.logs.materialize(self.state.log_directory, mode="intermediate")
            if self.state.maximum_triggers is not None and self.state.num_triggers >= self.state.maximum_triggers:
                logger.info("Exiting replay loop due to trigger limit.")
                break

    @register_stage(replay_data_pipeline, PipelineStage.REPLAY_DATA_DONE, next=None)
    def replay_data_done(self) -> None:
        self.state.pipeline_status_queue.put(pipeline_stage_msg(
            PipelineStage.REPLAY_DATA_DONE, MsgType.DATASET, dataset_submsg(self.state.pipeline_config.data.dataset_id)
        ))
        return None  # finish replay_data subpipeline

    # Wait for new data
    
    @register_stage(wait_for_new_data_pipeline, PipelineStage.FETCH_NEW_DATA, next=None)
    def fetch_new_data(self) -> PipelineStage | None:
        last_timestamp = self.state.start_timestamp
        dataset_id = self.state.pipeline_config.data.dataset_id
        continue_running = True
        
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.FETCH_NEW_DATA, MsgType.DATASET, dataset_submsg(dataset_id))
        )

        trigger_occurred = False
        largest_keys = set()
        for new_data, _ in self.grpc.get_new_data_since(dataset_id, last_timestamp):
            # Since get_new_data_since is inclusive, we need to filter out the keys
            # we have already processed in the previous get_new_data_since request
            new_data = [
                (key, timestamp, label)
                for (key, timestamp, label) in new_data
                if key not in self.state.previous_largest_keys
            ]
            last_timestamp = (
                max((timestamp for (_, timestamp, _) in new_data)) if len(new_data) > 0 else last_timestamp
            )

            # Remember all data points with last_timestamp so we do not process them again in the next iteration
            # We use a set to have a O(1) check in the line above.
            largest_keys.update({key for (key, timestamp, _) in new_data if timestamp == last_timestamp})

            # process batch
            self.execute(new_data_pipeline, PipelineStage.EVALUATE_TRIGGER_ON_BATCH)

            if self.state.previous_new_data_had_trigger:
                trigger_occurred = True

            if self.state.maximum_triggers is not None and self.state.num_triggers >= self.state.maximum_triggers:
                continue_running = False

        self.state.previous_largest_keys = largest_keys
        if not trigger_occurred and continue_running:
            return PipelineStage.WAIT_FOR_NEW_DATA  # another iteration
        
        return None  # finish wait_for_new_data subpipeline
        
    @register_stage(wait_for_new_data_pipeline, PipelineStage.WAIT_FOR_NEW_DATA, next=PipelineStage.FETCH_NEW_DATA)
    def wait_for_new_data(self) -> None:
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.WAIT_FOR_NEW_DATA, MsgType.DATASET, dataset_submsg(
                self.state.pipeline_config.data.dataset_id
            ))
        )
        sleep(2)

    # Process new data
    
    @register_stage(new_data_pipeline, PipelineStage.HANDLE_NEW_DATA, next=PipelineStage.NEW_DATA_HANDLED)
    def handle_new_data(self, new_data: list[tuple[int, int, int]]) -> None:
        """Handle new data during experiments or actual pipeline execution.
        
        We partition `new_data` into batches of `selector_batch_size` to reduce selector latency in case of a trigger.
        If a data point within a batch causes a trigger,
        we inform the selector about all data points including that data point.
        Otherwise, the selector is informed
        """
        logger.info(f"Received {len(new_data)} new data points. Handling batches.")
        new_data.sort(key=lambda tup: tup[1])
        any_training_triggered = False
        new_data_len = len(new_data)
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.HANDLE_NEW_DATA,
                MsgType.COUNTER,
                counter_submsg(CounterAction.CREATE, {"new_data_len": new_data_len})
            )
        )

        for i in range(0, new_data_len, self.state.selector_batch_size):
            batch = new_data[i : i + self.state.selector_batch_size]
            batch_size = self.state.selector_batch_size \
                if i + self.state.selector_batch_size < new_data_len else new_data_len - i
            self.state.pipeline_status_queue.put(
                pipeline_stage_msg(
                    PipelineStage.HANDLE_NEW_DATA,
                    MsgType.COUNTER,
                    counter_submsg(CounterAction.UPDATE, {"batch_size": batch_size})
                )
            )

            # execute batch subpipeline
            self.current_batch = batch
            self.execute(new_data_batch_pipeline, PipelineStage.EVALUATE_TRIGGER_ON_BATCH)
            any_training_triggered = any_training_triggered or self.state.previous_batch_had_trigger
            
            if self.state.maximum_triggers is not None and self.state.num_triggers >= self.state.maximum_triggers:
                logger.info(f"Reached trigger limit ({self.state.maximum_triggers}), exiting.")
                break

        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.NEW_DATA_HANDLED, MsgType.COUNTER, counter_submsg(CounterAction.CLOSE))
        )
        self.state.previous_new_data_had_trigger = any_training_triggered

    # Process new data batch
    
    @register_stage(new_data_batch_pipeline, PipelineStage.EVALUATE_TRIGGER_ON_BATCH)
    def evaluate_trigger_policies(self) -> PipelineStage:
        """Evaluate trigger policy and inform selector."""
        batch = self.state.new_data
        
        # Evaluate trigger policy
        start = datetime.datetime.now()
        triggering_indices = self.trigger.inform(batch)
        end = datetime.datetime.now()

        num_triggers = len(triggering_indices)
        self.logs.supervisor.num_trigger += num_triggers
        self.logs.supervisor.trigger_batch_times.append(
            TriggerBatchTimeLog(batch_size=len(batch), num_triggers=num_triggers, start=start, end=end)
        )
        
        trigger_occurred = num_triggers > 0
        
        # persist state
        self.state.num_triggers = num_triggers
        self.state.previous_batch_had_trigger = trigger_occurred
        self.state.current_batch_triggering_indices = triggering_indices
        
        if trigger_occurred:
            return PipelineStage.EXECUTE_TRIGGERS_WITHIN_BATCH
        
        return PipelineStage.INFORM_SELECTOR_NO_TRIGGER
    
    @register_stage(new_data_batch_pipeline, PipelineStage.INFORM_SELECTOR_NO_TRIGGER, next=None)
    def inform_selector_no_trigger(self) -> None:
        batch = self.state.new_data

        start = datetime.datetime.now()
        selector_log = self.grpc.inform_selector(self.state.pipeline_id, batch)
        end = datetime.datetime.now()
        self.logs.supervisor.selector_informs.append(SelectorInformLog(selector_log=selector_log, start=start, end=end))
        return None  # end of `new_data_batch_pipeline`
    
    @register_stage(new_data_batch_pipeline, PipelineStage.EXECUTE_TRIGGERS_WITHIN_BATCH)
    def execute_triggers_within_batch(self) -> PipelineStage | None:
        """Evaluate trigger policy, start training after trigger and inform selector."""
        logger.info(f"There are {self.state.num_triggers} triggers in this batch.")
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.EXECUTE_TRIGGERS_WITHIN_BATCH, MsgType.GENERAL)
        )
        
        # unpack state
        batch = self.state.new_data
        triggering_indices = self.state.current_batch_triggering_indices
        assert triggering_indices
        
        self.state.current_batch_previous_trigger_idx = 0
        logger.info("Handling triggers within batch.")

        for triggering_idx in triggering_indices:
            self.state.current_batch_next_trigger_id = triggering_idx
            self.execute(execute_trigger, PipelineStage.INFORM_SELECTOR_AND_TRIGGER)
            
            self.state.num_triggers = self.state.num_triggers + 1
            if self.state.maximum_triggers is not None and self.state.num_triggers >= self.state.maximum_triggers:
                break

        # If no other trigger is coming in this batch,
        # we have to inform the Selector about the remaining data in this batch.
        remaining_data = batch[triggering_idx + 1 :]
        logger.info(f"There are {len(remaining_data)} data points remaining after the trigger.")

        self.state.current_batch_remaining_data = remaining_data
        if len(remaining_data) > 0:
            return PipelineStage.INFORM_SELECTOR_REMAINING_DATA

        return None

    @register_stage(new_data_pipeline, PipelineStage.INFORM_SELECTOR_REMAINING_DATA, next=None)
    def inform_selector_remaining_data(self) -> None:
        """No trigger occurred in the batch, inform selector about remaining data."""
        
        remaining_data = self.state.current_batch_remaining_data
        trigger_id = self.state.current_batch_next_trigger_id
        
        # These data points will be included in the next trigger
        # because we inform the Selector about them,
        # just like other batches with no trigger at all are included.
        self.state.pipeline_status_queue.put(pipeline_stage_msg(
            PipelineStage.INFORM_SELECTOR_REMAINING_DATA, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id)
        ))

        start = datetime.datetime.now()
        selector_log = self.grpc.inform_selector(self.state.pipeline_id, remaining_data)
        self.logs.supervisor.selector_informs.append(
            SelectorInformLog(selector_log=selector_log, start=start, end=datetime.datetime.now())
        )
        self.logs.materialize()
        
        return None  # end of `new_data_pipeline`


    # Execute trigger within batch

    @register_stage(execute_trigger, PipelineStage.INFORM_SELECTOR_AND_TRIGGER, next=PipelineStage.INFORM_SELECTOR_REMAINING_DATA)
    def inform_selector_and_trigger(self) -> None:
        # unpack state
        batch = self.state.new_data
        trigger_idx = self.state.current_batch_previous_trigger_idx
        
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.INFORM_SELECTOR_AND_TRIGGER, MsgType.GENERAL)
        )
        triggering_data = batch[trigger_idx : trigger_idx + 1]
        self.state.current_batch_previous_trigger_idx = self.state.current_batch_next_trigger_id + 1

        # This call informs the selector about the data until (and including)
        # the data point that caused the trigger and then also notifies it about the triggering.
        # This means the next training call on trigger_id will guarantee
        # that all data until that point has been processed by the selector.
        start = datetime.datetime.now()
        trigger_id, selector_log = self.grpc.inform_selector_and_trigger(self.state.pipeline_id, triggering_data)
        end = datetime.datetime.now()
        self.logs.supervisor.triggers[trigger_id] = SelectorInformLog(
            selector_log=selector_log, start=start, end=end
        )
        self.logs.materialize()
        
        self.state.current_trigger_id = trigger_id
    
    @register_stage(execute_trigger, PipelineStage.RUN_TRAINING)
    def run_training_pipeline(self) -> None:
        # unpack state
        trigger_id = self.state.current_trigger_id
        
        num_samples_in_trigger = self.grpc.get_number_of_samples(self.state.pipeline_id, trigger_id)
        if num_samples_in_trigger > 0:
            # Blocks until training is done.
            self.execute(training_pipeline, PipelineStage.RUN_TRAINING)

            self.state.pipeline_status_queue.put(pipeline_stage_msg(
                PipelineStage.EXECUTE_TRIGGERS_WITHIN_BATCH, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id)
            ))

        else:
            logger.info(f"Skipping training on empty trigger {trigger_id}]")

    # Training

    @register_stage(training_pipeline, PipelineStage.RUN_TRAINING, next=PipelineStage.WAIT_FOR_TRAINING_COMPLETION)
    def run_training(self) -> None:
        """Run training for trigger on GPU and block until done."""
        trigger_id = self.state.current_batch_next_trigger_id
        
        assert self.state.pipeline_id is not None, "_run_training called without a registered pipeline."
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.RUN_TRAINING, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )
        logger.info(f"Running training for trigger {trigger_id}")

        num_samples_to_pass_per_trigger = self.state.pipeline_config.training.num_samples_to_pass or []
        current_trigger_index = len(self.state.triggers)
        if current_trigger_index <= len(num_samples_to_pass_per_trigger) - 1:
            num_samples_to_pass = num_samples_to_pass_per_trigger[current_trigger_index]
        else:
            num_samples_to_pass = None

        self.logs.supervisor.triggers[trigger_id] = TriggerLog(
            trainer_log={},
            start=datetime.datetime.now(),
            end=None,
        )
        self.state.current_training_id = self.grpc.start_training(
            self.state.pipeline_id, trigger_id, self.state.pipeline_config, self.state.previous_model_id, num_samples_to_pass
        )
            
    @register_stage(training_pipeline, PipelineStage.WAIT_FOR_TRAINING_COMPLETION, next=PipelineStage.TRAINING_COMPLETED)
    def wait_for_training_completion(self) -> None:
        trigger_id = self.state.current_batch_next_trigger_id

        self.grpc.wait_for_training_completion(self.state.current_training_id, self.state.pipeline_id, trigger_id)

        # report training completion
        self.logs.supervisor.triggers[trigger_id].end_training = datetime.datetime.now()
    
    @register_stage(training_pipeline, PipelineStage.TRAINING_COMPLETED, next=PipelineStage.STORE_TRAINED_MODEL)
    def training_completed(self) -> None:
        trigger_id = self.state.current_batch_next_trigger_id
        
        self.state.pipeline_status_queue.put(pipeline_stage_msg(
            PipelineStage.TRAINING_COMPLETED, MsgType.ID, id_submsg(IdType.TRAINING, self.state.current_training_id), True
        ))
        logger.debug("Training completed")

        
        if trigger_id not in self.logs.supervisor.triggers:
            self.logs.supervisor.triggers[trigger_id] = {}  # can happen in tests
    
    @register_stage(training_pipeline, PipelineStage.STORE_TRAINED_MODEL, next=None)
    def store_trained_model(self) -> None:
        trigger_id = self.state.current_batch_next_trigger_id

        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.STORE_TRAINED_MODEL, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )
        
        trigger_id = self.state.current_batch_next_trigger_id
        
        # We store the trained model for evaluation in any case.
        self.logs.supervisor.triggers[trigger_id].start_store_model = datetime.datetime.now()
        model_id = self.grpc.store_trained_model(self.state.current_training_id)
        self.logs.supervisor.triggers[trigger_id].end_store_model = datetime.datetime.now()

        # Only if the pipeline actually wants to continue the training on it, we set previous model.
        if self.state.pipeline_config.training["use_previous_model"]:
            self.state.previous_model_id = model_id

        self.state.trained_models.append(model_id)
        self.state.triggers.append(trigger_id)

        return None  # End of training subpipeline

    # Evaluation
    
    @register_stage(evaluation_pipeline, PipelineStage.EVALUATE, next=PipelineStage.WAIT_FOR_EVALUATION_COMPLETION)
    def evaluate(self) -> None:
        model_id = self.state.previous_model_id        
        trigger_id = self.state.current_batch_next_trigger_id

        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.EVALUATE, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )
        
        # TODO(#300) Add evaluator to pipeline log
        self.state.current_batch_evaluations = self.grpc.start_evaluation(model_id, self.state.pipeline_config)
        
    @register_stage(
        evaluation_pipeline, PipelineStage.WAIT_FOR_EVALUATION_COMPLETION, next=PipelineStage.EVALUATION_COMPLETED
    )
    def wait_for_evaluation_completed(self) -> None:
        self.grpc.wait_for_evaluation_completion(self.state.current_training_id, self.state.current_batch_evaluations)
        
    @register_stage(
        evaluation_pipeline, PipelineStage.EVALUATION_COMPLETED, next=PipelineStage.STORE_EVALUATION_RESULTS
    )
    def evaluation_completed(self) -> None:
        pass # nothing to do
    
    @register_stage(
        evaluation_pipeline, PipelineStage.STORE_EVALUATION_RESULTS, next=PipelineStage.EXECUTE_TRIGGERS_WITHIN_BATCH
    )
    def store_evaluation_results(self) -> None:
        trigger_id = self.state.current_batch_next_trigger_id
        self.state.pipeline_status_queue.put(
            pipeline_stage_msg(PipelineStage.STORE_EVALUATION_RESULTS, MsgType.ID, id_submsg(IdType.TRIGGER, trigger_id))
        )

        writer_names: set[str] = set(self.state.pipeline_config["evaluation"]["result_writers"])
        writers = [self._init_evaluation_writer(name, trigger_id) for name in writer_names]
        self.grpc.store_evaluation_results(writers, self.state.current_batch_evaluations)

    # Teardown

    @register_stage(main_pipeline, PipelineStage.DONE, next=PipelineStage.EXIT)
    def done(self) -> None:
        self.state.pipeline_status_queue.put(pipeline_stage_msg(PipelineStage.DONE, MsgType.GENERAL))
        self.logs.materialize(self.state.log_directory, mode="final")
    
    @register_stage(main_pipeline, PipelineStage.EXIT, next=None)
    def exit(self) -> None:
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

    def _init_evaluation_writer(self, name: str, trigger_id: int) -> LogResultWriter:
        return self.state.supervisor_supported_eval_result_writers[name](self.state.pipeline_id, trigger_id, self.state.eval_directory)

    def _shutdown_trainer(self) -> None:
        if self.state.current_training_id is not None:
            self.grpc.stop_training_at_trainer_server(self.state.current_training_id)


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
