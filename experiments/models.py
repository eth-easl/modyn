from dataclasses import dataclass

from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.metrics import MetricConfig


@dataclass
class Experiment:
    name: str
    eval_handlers: list[EvalHandlerConfig]
    time_trigger_schedules: list[str | int]  # in years
    data_amount_triggers: list[int]  # in num samples
    drift_detection_intervals: list[int]  # every interval configures one pipeline

    # list of metric configuration, every list item will yield a new pipeline for every drift_detection_intervals
    drift_trigger_metrics: list[dict[str, MetricConfig]]

    gpu_device: str

    # optional:
    warmup_until: int | None = None  # Don't start the first training until warmup_until
