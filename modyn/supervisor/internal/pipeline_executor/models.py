from __future__ import annotations

import datetime
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.config.schema.config import ModynConfig
from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.utils.evaluation_status_reporter import (
    EvaluationStatusReporter,
)


class PipelineOptions(BaseModel):
    """
    Wrapped cli argument bundle for the pipeline executor.
    """
    
    start_timestamp: int
    pipeline_id: int
    modyn_config: ModynConfig
    pipeline_config: ModynPipelineConfig
    eval_directory: Path
    supervisor_supported_eval_result_writers: dict
    exception_queue: mp.Queue
    pipeline_status_queue: mp.Queue
    training_status_queue: mp.Queue
    eval_status_queue: mp.Queue
    start_replay_at: int | None = Field(None)
    stop_replay_at: int | None = Field(None)
    maximum_triggers: int | None = Field(None)
    
    @property
    def log_directory(self) -> Path:
        return self.eval_directory

    @property
    def experiment_mode(self) -> bool:
        return self.start_replay_at is not None

    @cached_property
    def selector_batch_size(self) -> int:
        dataset_id = self.pipeline_config.data.dataset_id

        for dataset in self.modyn_config.storage.datasets:
            if dataset.name == dataset_id:
                return dataset.selector_batch_size
        
        assert False, f"Dataset with id {dataset_id} not found in storage configuration."
            
        

    @model_validator
    @classmethod
    def validate_replay(cls, v: PipelineOptions) -> PipelineOptions:
        if v.start_replay_at is None and v.stop_replay_at is not None:
            raise ValueError("`stop_replay_at` can only be used in conjunction with `start_replay_at`.")
        return v


# TODO: pipeline hierarchy abstraction e.g. main pipeline, new data batch pipeline, eval pipeline, ...

class PipelineBatchState(BaseModel):
    """Pipeline artifacts shared across stages during the processing of one batch."""
    data: list[tuple[int, int, int]] = Field(default_factory=list)
    num_triggers: int = Field(0)
    triggering_indices: list[int] = Field(0)
    previous_trigger_idx: int = Field(-1)
    next_trigger_id: int = Field(-1)
    remaining_data: list[tuple[int, int, int]] = Field(default_factory=list)
    evaluations: dict[int, EvaluationStatusReporter] = Field(default_factory=dict)
    trigger_id: int = Field(-1)

class ExecutionState(PipelineOptions):
    """Represent the state of the pipeline executor including artifacts of pipeline stages."""
        
    grpc: GRPCHandler

    num_triggers: int = Field(0)
    trained_models: list[int] = []
    triggers: list[int] = []
    
    sw = Field(default_factory=Stopwatch)
    
    # Internal state for pipeline steps
    
    # RUN_TRAINING, STORE_TRAINED_MODEL
    previous_model_id: Optional[int] = Field(None)
    training_id: Optional[int] = None
    
    # PipelineStage.REPLAY_DATA
    new_data: list[tuple[int, int, int]] = Field(default_factory=list)
    """Stores the new unprocessed data to be processed in `HANDLE_NEW_DATA`."""
    
    # PipelineStage.FETCH_NEW_DATA
    previous_largest_keys: set[int] = Field(default_factory=set)
    
    # new_data_batch_pipeline
    batch: PipelineBatchState = Field(PipelineBatchState())
    
    # INFORM_SELECTOR_AND_TRIGGER
    
    previous_new_data_had_trigger: bool = Field(False)
    previous_batch_had_trigger: bool = Field(False)
    
    
    @model_validator
    @classmethod
    def validate_mode(cls, v: ExecutionState) -> ExecutionState:
        if (v.experiment is None) != (v.replay_data is None):
            raise ValueError("`experiment` and `replay_data` must be specified together.")
        return v


@dataclass
class RegisteredStage:
    """Represent a registered pipeline stage that includes a callable function and the next stage."""
    
    stage: PipelineStage
    func: Callable
    next: PipelineStage | None = None
    """If next stage if None, the next stage will be decided by the return value of the current stage's `func`.
    
    If both are None, the pipeline ends.
    """
    
    logging: bool = True
    
class PipelineLogsConfigs(BaseModel):
    """
    Wrapped logs for the pipeline executor.
    """
    
    modyn_config: ModynConfig
    pipeline_config: PipelineLogsConfigs


class StageRunLog(BaseModel):
    id: str
    start: datetime.datetime
    end: datetime.datetime
    # TODO: start, end, duration, status

class NewDataRequestLog(BaseModel):
    time: datetime.datetime
    num_items: int
    
class TriggerLog(BaseModel):
    trainer_log: dict[str, Any]
    start_training: datetime.datetime
    end_training: datetime.datetime | None = Field(None)
    start_store_model: datetime.datetime | None = Field(None)
    end_store_model: datetime.datetime | None = Field(None)


# TODO: log sample time...
class TriggerBatchTimeLog(BaseModel):
    batch_size: int
    num_triggers: int
    start: datetime.datetime
    end: datetime.datetime
    

class SelectorInformLog(BaseModel):
    selector_log: dict[str, Any]
    start: datetime.datetime
    end: datetime.datetime


class SupervisorLogs(BaseModel):
    stage_runs: list[StageRunLog] = Field(default_factory=dict)
    
    num_trigger: int = Field(0, description="Total number of triggers, including multiple triggers per batch.")
    new_data_requests: list[NewDataRequestLog] = Field(default_factory=list)
    
    triggers: dict[int, TriggerLog] = Field(default_factory=list)
    
    trigger_batch_times: list[TriggerBatchTimeLog] = Field(default_factory=list)
    selector_informs: list[SelectorInformLog] = Field(default_factory=list)
    
    # TODO dataframe conversions
    
    def clear(self) -> None:
        self.stage_runs.clear()
        
    def merge(self, logs: list[StageRunLog]) -> SupervisorLogs:
        self.stage_runs = self.stage_runs + logs
        return self


class PipelineLogs(BaseModel):
    """
    Wrapped logs for the pipeline executor.
    """
    
    pipeline_id: int
    
    config: PipelineLogsConfigs
    supervisor: SupervisorLogs

    experiment: bool
    replay_data: bool
    
    def materialize(self, log_dir_path: Path, mode: Literal["initial", "increment", "final"] = "increment") -> None:
        """Materialize the logs to log files.
        
        If run with pytest, log_file_path and mode will be ignored.
        
        Args:
            log_dir_path: The path to the log file.
            mode: The mode to materialize the logs. Initial will output static fields, increment will output only 
                new items since the last materialization, and final will merge all logs.
        """
        
        if "PYTEST_CURRENT_TEST" in os.environ:
            self.model_validate_json()  # Enforce serialization to catch issues
            return  # But don't actually store in tests

        filename = f"pipeline_{self.pipeline_id}"
        if mode == "initial":
            with open(log_dir_path / (filename + ".log"), "w", encoding="utf-8") as logfile:
                logfile.write(self.model_dump_json(indent=2))
            return

        if mode == "increment":
            filepath = log_dir_path / (f"{filename}_supervisor_part_{datetime.datetime.now().isoformat()}.log")
            with open(filepath, "w", encoding="utf-8") as logfile:
                logfile.write(json.dumps(self.supervisor.stage_runs, indent=2))
            
            self.supervisor.clear()
            return 
            
        # final: merge increment logs
        supervisor_files = sorted(log_dir_path.glob(f"{filename}_supervisor_part_*.log"))
        partial_supervisor_logs = [
            StageRunLog.model_validate(stage_run)
            for file in supervisor_files
            for stage_run in json.loads(file.read_text(encoding="utf-8"))
        ]
        self.supervisor = self.supervisor.merge(partial_supervisor_logs)
        
        with open(log_dir_path / (filename + ".log"), "w", encoding="utf-8") as logfile:
            logfile.write(self.model_dump_json(indent=2))
    
        self.supervisor.clear()

