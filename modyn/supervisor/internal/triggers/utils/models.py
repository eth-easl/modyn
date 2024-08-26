from typing import Annotated, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel
from modyn.config.schema.pipeline.trigger.drift.result import MetricResult


class DriftTriggerEvalLog(ModynBaseModel):
    type: Literal["drift"] = Field("drift")
    detection_interval: tuple[int, int]  # timestamps of the current detection interval
    reference_interval: tuple[int | None, int | None] = Field((None, None))  # timestamps of the reference interval
    triggered: bool
    trigger_index: int | None = Field(None)
    data_points: int = Field(0)
    drift_results: dict[str, MetricResult] = Field(default_factory=dict)
    is_warmup: bool = Field(False)


class EnsembleTriggerEvalLog(ModynBaseModel):
    type: Literal["ensemble"] = Field("ensemble")
    triggered: bool
    trigger_index: int | None = Field(None)  # in inform(..) batch
    evaluation_interval: tuple[tuple[int, int], tuple[int, int]]
    subtrigger_decisions: dict[str, bool] = Field(
        default_factory=dict,
        description="The policy decisions for the evaluation interval for different metrics keyed by their name.",
    )


class PerformanceTriggerEvalLog(ModynBaseModel):
    type: Literal["performance"] = Field("performance")
    triggered: bool
    trigger_index: int | None = Field(None)
    evaluation_interval: tuple[int, int]  # timestamps of the current detection interval
    num_samples: int = Field(0, description="The number of data points in the evaluation interval.")
    num_misclassifications: int = Field(0, description="The number of misclassifications in the evaluation interval.")
    evaluation_scores: dict[str, float] = Field(
        default_factory=dict,
        description="The evaluation scores for the evaluation interval for different metrics keyed by their name.",
    )
    policy_decisions: dict[str, bool] = Field(
        default_factory=dict,
        description="The policy decisions for the evaluation interval for different metrics keyed by their name.",
    )


TriggerEvalLog = Annotated[
    DriftTriggerEvalLog | EnsembleTriggerEvalLog | PerformanceTriggerEvalLog,
    Field(discriminator="type"),
]


class TriggerPolicyEvaluationLog(ModynBaseModel):
    evaluations: list[TriggerEvalLog] = Field(
        default_factory=list,
        description="The results of the trigger policy evaluation.",
    )
