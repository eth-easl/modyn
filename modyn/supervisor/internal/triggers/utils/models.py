from typing import Annotated, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel
from modyn.config.schema.pipeline.trigger.drift.result import MetricResult


class _BaseTriggerEvalLog(ModynBaseModel):
    triggered: bool
    trigger_index: int | None = Field(None)
    num_samples: int = Field(0, description="The number of data points in the evaluation interval.")
    evaluation_interval: tuple[int, int]  # timestamps of the current detection batch


class DriftTriggerEvalLog(_BaseTriggerEvalLog):
    type: Literal["drift"] = Field("drift")
    id_model: int | None = Field(None, description="The model ID used for the embedding generation.")
    detection_interval: tuple[int, int]  # timestamps of the current detection interval
    reference_interval: tuple[int | None, int | None] = Field((None, None))  # timestamps of the reference interval
    drift_results: dict[str, MetricResult] = Field(default_factory=dict)
    is_warmup: bool = Field(False)


class EnsembleTriggerEvalLog(_BaseTriggerEvalLog):
    type: Literal["ensemble"] = Field("ensemble")
    subtrigger_decisions: dict[str, bool] = Field(
        default_factory=dict,
        description="The policy decisions for the evaluation interval for different metrics keyed by their name.",
    )


class PerformanceTriggerEvalLog(_BaseTriggerEvalLog):
    type: Literal["performance"] = Field("performance")
    id_model: int | None = Field(None, description="The model ID of the model that was evaluated.")
    num_misclassifications: int = Field(0, description="The number of misclassifications in the evaluation interval.")
    evaluation_scores: dict[str, float] = Field(
        default_factory=dict,
        description="The evaluation scores for the evaluation interval for different metrics keyed by their name.",
    )
    policy_decisions: dict[str, bool] = Field(
        default_factory=dict,
        description="The policy decisions for the evaluation interval for different metrics keyed by their name.",
    )


class CostAwareTriggerEvalLog(_BaseTriggerEvalLog):
    type: Literal["cost_aware"] = Field("cost_aware")
    regret_metric: float = Field(description="The regret metric value before making the decision.")
    regret_log: dict = Field(description="Additional logs about the regret metric computation.")
    traintime_estimate: float = Field(description="The forecasted training time for the next model.")
    regret_in_traintime_unit: float = Field(description="The regret metric value in training time units.")


TriggerEvalLog = Annotated[
    CostAwareTriggerEvalLog | DriftTriggerEvalLog | EnsembleTriggerEvalLog | PerformanceTriggerEvalLog,
    Field(discriminator="type"),
]


class TriggerPolicyEvaluationLog(ModynBaseModel):
    evaluations: list[TriggerEvalLog] = Field(
        default_factory=list,
        description="The results of the trigger policy evaluation.",
    )
