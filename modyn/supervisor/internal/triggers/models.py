from typing import Annotated, Literal, Union

from modyn.config.schema.base_model import ModynBaseModel
from modyn.config.schema.pipeline.trigger.drift.result import MetricResult
from pydantic import Field


class DriftTriggerEvalLog(ModynBaseModel):
    type: Literal["drift"] = Field("drift")
    detection_interval: tuple[int, int]  # timestamps of the current detection interval
    reference_interval: tuple[int | None, int | None] = Field((None, None))  # timestamps of the reference interval
    triggered: bool
    trigger_index: int | None = Field(None)
    data_points: int = Field(0)
    drift_results: dict[str, MetricResult] = Field(default_factory=dict)
    is_warmup: bool = Field(False)


TriggerEvalLog = Annotated[Union[DriftTriggerEvalLog], Field(discriminator="type")]


class TriggerPolicyEvaluationLog(ModynBaseModel):
    evaluations: list[TriggerEvalLog] = Field(
        default_factory=list, description="The results of the trigger policy evaluation."
    )
