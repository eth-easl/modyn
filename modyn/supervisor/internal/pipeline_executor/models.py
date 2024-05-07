from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp

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
