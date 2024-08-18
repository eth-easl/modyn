from dataclasses import dataclass

from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.metrics import MetricConfig
from modyn.config.schema.pipeline.trigger.drift.config import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.data_amount import DataAmountTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.time import TimeTriggerConfig


@dataclass
class Experiment:
    name: str
    eval_handlers: list[EvalHandlerConfig]

    time_triggers: dict[str, TimeTriggerConfig]
    data_amount_triggers: dict[str, DataAmountTriggerConfig]
    drift_detection_triggers: dict[str, DataDriftTriggerConfig]

    gpu_device: str

    # optional:
    warmup_until: int | None = None  # Don't start the first training until warmup_until
