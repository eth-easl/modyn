from dataclasses import dataclass
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig


@dataclass
class Experiment:
    name: str
    eval_handlers: list[EvalHandlerConfig]
    time_trigger_schedules: list[int]  # in years
    data_amount_triggers: list[int]  # in num samples
    drift_triggers: list[tuple[int, float]]  # interval, threshold
    gpu_device: str
