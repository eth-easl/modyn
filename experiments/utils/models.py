from dataclasses import dataclass, field

from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.trigger.cost.cost import CostTriggerConfig
from modyn.config.schema.pipeline.trigger.drift.config import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.ensemble import EnsembleTriggerConfig
from modyn.config.schema.pipeline.trigger.performance.performance import PerformanceTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.data_amount import DataAmountTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.time import TimeTriggerConfig


@dataclass
class Experiment:
    name: str
    eval_handlers: list[EvalHandlerConfig]

    time_triggers: dict[str, TimeTriggerConfig] = field(default_factory=dict)
    data_amount_triggers: dict[str, DataAmountTriggerConfig] = field(default_factory=dict)
    drift_detection_triggers: dict[str, DataDriftTriggerConfig] = field(default_factory=dict)
    performance_triggers: dict[str, PerformanceTriggerConfig] = field(default_factory=dict)
    cost_triggers: dict[str, CostTriggerConfig] = field(default_factory=dict)
    ensemble_triggers: dict[str, EnsembleTriggerConfig] = field(default_factory=dict)

    gpu_device: str = "cuda:0"

    # optional:
    warmup_until: int | None = None  # Don't start the first training until warmup_until

    seed: int = 0
