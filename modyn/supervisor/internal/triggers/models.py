from typing import Annotated, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel
from modyn.config.schema.pipeline.trigger.drift.result import MetricResult


class DriftTriggerEvalLog(ModynBaseModel):
    type: Literal["drift"] = Field("drift")
    detection_idx_start: int
    detection_idx_end: int
    triggered: bool
    trigger_index: int | None = Field(None)
    data_points: int = Field(0)
    drift_results: dict[str, MetricResult] = Field(default_factory=dict)


TriggerEvalLog = Annotated[DriftTriggerEvalLog, Field(discriminator="type")]


class TriggerPolicyEvaluationLog(ModynBaseModel):
    evaluations: list[TriggerEvalLog] = Field(
        default_factory=list, description="The results of the trigger policy evaluation."
    )
