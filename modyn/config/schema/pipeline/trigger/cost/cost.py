"""Cost aware triggers are evaluate the trade-off between the cost of a trigger
(e.g. measured by training machine time) and the benefit of the trigger (e.g.
measured by a regret metric like data incorporation latency or avoidable
misclassification incorporation latency)."""

from typing import Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel

from ..performance.performance import _InternalPerformanceTriggerConfig


class _CostTriggerConfig(ModynBaseModel):
    """Base class for cost aware trigger policies."""

    cost_tracking_window_size: int = Field(
        description="How many trigger into the past should be considered for the linear cost w.r.t. data amount model."
    )


class DataIncorporationLatencyCostTriggerConfig(_CostTriggerConfig):
    """Cost aware trigger policy configuration that uses the data incorporation
    latency as a regret metric.

    While no trigger occurs samples are cumulated into to a number of un-integrated samples over time curve.
    Rather than using this metric directly, we build an area-under-the-curve metric.
    The area under the un-integrated samples curve measures the time samples have spent in the incorporation queue.
    `incorporation_delay_per_training_second` is as a conversion rate between cost budget (training time) and regret
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


class AvoidableMisclassificationCostTriggerConfig(_CostTriggerConfig, _InternalPerformanceTriggerConfig):
    """Cost aware trigger policy configuration that using the number of
    avoidable misclassifications integration latency as a regret metric.

    This policy can be seen a combination of data incorporation latency based cost triggers and performance aware
    triggers.

    `avoidable_misclassification_latency_per_training_second` servers as a conversion rate between budget (training time)
    and regret metric (misclassifications).

    When a the regret metric exceeds the budget, a trigger is fired.

    Like for performance aware triggers the same set of `decision_criteria` as `PerformanceTriggerConfig`
    but implicitly adds a cost criterion to the list.

    Not only evaluates data density and model performance but also
    consider the cost of a trigger both in terms of wall clock time and
    number of triggers.

    We use the `_InternalPerformanceTriggerConfig` base class as we cannot override the `id` field of
    `PerformanceTriggerConfig`.
    """

    id: Literal["AvoidableMisclassificationCostTrigger"] = Field("AvoidableMisclassificationCost")

    # Conversion rate between budget (training time) and regret metric (misclassifications)
    avoidable_misclassification_latency_per_training_second: float = Field(
        description="How many seconds of unaddressed avoidable misclassifications are we willing to accept per second of training time saved."
    )
