"""Cost aware triggers are evaluate the trade-off between the cost of a trigger
(e.g. measured by training machine time) and the benefit of the trigger (e.g.
measured by a regret metric like data incorporation latency or avoidable
misclassification incorporation latency)."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from modyn.config.schema.pipeline.trigger.common.batched import BatchedTriggerConfig

from ..performance.criterion import _NumberAvoidableMisclassificationCriterion
from ..performance.performance import _InternalPerformanceTriggerConfig


class _CostTriggerConfig(BatchedTriggerConfig):
    """Base class for cost aware trigger policies."""

    cost_tracking_window_size: int = Field(
        1000,
        description="How many trigger into the past should be considered for the linear cost w.r.t. data amount model.",
    )


class DataIncorporationLatencyCostTriggerConfig(_CostTriggerConfig):
    """Cost aware trigger policy configuration that uses the data incorporation
    latency as a regret metric.

    While no trigger occurs samples are cumulated into to a number of un-integrated samples over time curve.
    Rather than using this metric directly, we build an area-under-the-curve metric.
    The area under the un-integrated samples curve measures the time samples have spent in the incorporation queue.

    As this policy operates the two metrics `time` (cost) and `incorporation_delay` (regret) we need
    a way to express the tradeoff between the two. A user e.g. has to specify how many seconds of training time he is
    willing to eradicate a certain amount of cumulative regret (here `incorporation delay`).

    `incorporation_delay_per_training_second` is this conversion rate between cost budget (training time) and regret
    metric (incorporation latency).

    When the cumulated regret (area under the curve) exceeds the training-time budget, a trigger is fired.
    """

    id: Literal["DataIncorporationLatencyCostTrigger"] = Field("DataIncorporationLatencyCostTrigger")

    # Conversion rate between budget (training time) and regret metric (misclassifications)
    incorporation_delay_per_training_second: float = Field(
        description=(
            "How many seconds of samples having spent in the incorporation queue are we willing to accept "
            "per second of training time saved."
        )
    )

    @property
    def conversion_factor(self) -> float:
        """Conversion factor between budget (training time) and regret metric
        (incorporation latency).

        Unifies names for easier downstream usage.
        """
        return self.incorporation_delay_per_training_second


class AvoidableMisclassificationCostTriggerConfig(
    _CostTriggerConfig,
    _InternalPerformanceTriggerConfig,
    _NumberAvoidableMisclassificationCriterion,
):
    """Cost aware trigger policy configuration that using the number of
    avoidable misclassifications integration latency as a regret metric.

    We suspect the cumulated number of misclassifications is very unstable and badly conditioned on user input as
    it's a linear function with respect to the amount of data. As the training cost is also linear with respect to the
    amount of data, this likely lead to a very brittle trigger policy.
    That's why we penalize every second that an avoidable misclassification remains unaddressed (no trigger).

    It's a result of combining this data incorporation latency idea with the static number of misclassification
    performance trigger. This policy can be seen a combination of data incorporation latency based cost triggers
    and performance aware triggers.

    As this policy operates the two metrics `time` (cost) and `misclassification_incorporation_latency` (regret) we need
    a way to express the tradeoff between the two. A user e.g. has to specify how many seconds of training time he is
    willing to eradicate a certain amount of cumulative regret (here `incorporation delay`).

    `avoidable_misclassification_latency_per_training_second` is this conversion rate between cost budget
    (training time) and regret metric (misclassifications).

    When a the regret metric exceeds the budget, a trigger is fired.

    Like for performance aware triggers the same set of `decision_criteria` as `PerformanceTriggerConfig`
    but implicitly adds a cost criterion to the list.

    Not only evaluates data density and model performance but also
    consider the cost of a trigger both in terms of wall clock time and
    number of triggers.

    We use the `_InternalPerformanceTriggerConfig` base class as we cannot override the `id` field of
    `PerformanceTriggerConfig`.
    """

    id: Literal["AvoidableMisclassificationCostTrigger"] = Field("AvoidableMisclassificationCostTrigger")

    # Conversion rate between budget (training time) and regret metric (misclassifications)
    avoidable_misclassification_latency_per_training_second: float = Field(
        description="How many seconds of unaddressed avoidable misclassifications are we willing to accept per second of training time saved."
    )

    @property
    def conversion_factor(self) -> float:
        """Conversion factor between budget (training time) and regret metric
        (avoidable misclassifications).

        Unifies names for easier downstream usage.
        """
        return self.avoidable_misclassification_latency_per_training_second

    @model_validator(mode="after")
    def warmup_policy_requirement(self) -> Self:
        """Assert whether the warmup policy is set when a metric needs
        calibration."""
        if self.warmup_policy is None:
            raise ValueError("A warmup policy is required for cost triggers.")
        return self


CostTriggerConfig = Annotated[
    AvoidableMisclassificationCostTriggerConfig | DataIncorporationLatencyCostTriggerConfig,
    Field(discriminator="id"),
]
