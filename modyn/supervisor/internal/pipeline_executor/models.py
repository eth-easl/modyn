from __future__ import annotations

import multiprocessing as mp
from dataclasses import Field, dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class PipelineOptions:
    """
    Wrapped cli argument bundle for the pipeline executor.
    """
    
    start_timestamp: int
    pipeline_id: int
    modyn_config: dict
    pipeline_config: dict
    eval_directory: str
    supervisor_supported_eval_result_writers: dict
    exception_queue: mp.Queue
    pipeline_status_queue: mp.Queue
    training_status_queue: mp.Queue
    eval_status_queue: mp.Queue
    start_replay_at: int | None = None
    stop_replay_at: int | None = None
    maximum_triggers: int | None = None
    evaluation_matrix: bool = False


class PipelineLogsConfigs(BaseModel):
    """
    Wrapped logs for the pipeline executor.
    """
    
    modyn_config: ModynConfig
    pipeline_config: PipelineConfig


class StageRuns(BaseModel):
    id: str
    # TODO: start, end, duration, status

class TriggerLog(BaseModel):
    pass


class SupervisorLogs(BaseModel):
    stage_runs: dict[str, Any] = Field(default_factory=dict)
    
    triggers: list[TriggerLog] = Field(default_factory=list)
    new_data_requests
    num_trigger
    trigger_batch_times
    selector_informs


class PipelineLogs(BaseModel):
    """
    Wrapped logs for the pipeline executor.
    """
    
    pipeline_id: int
    config: PipelineLogsConfigs
    supervisor: SupervisorLogs
