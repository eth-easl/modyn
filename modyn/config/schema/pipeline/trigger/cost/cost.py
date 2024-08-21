"""Cost aware triggers are evaluate the trade-off between the cost of a trigger
(e.g. measured by training machine time) and the benefit of the trigger (e.g.
measured by a regret metric like data incorporation latency or avoidable
misclassification incorporation latency)."""

from typing import Annotated, Literal

from pydantic import Field

from modyn.config.schema.pipeline.trigger.common.mixins import (
    TriggerEvaluationIntervalMixin,
)

from ..performance.performance import _InternalPerformanceTriggerConfig


class _CostTriggerConfig(TriggerEvaluationIntervalMixin):
    """Base class for cost aware trigger policies."""

    cost_tracking_window_size: int = Field(
        1000,
        description="How many trigger into the past should be considered for the linear cost w.r.t. data amount model.",
    )


class DataIncorporationLatencyCostTriggerConfig(_CostTriggerConfig):
    """Cost aware trigger policy configuration that uses the data incorporation
    latency as a regret metric.

    While no trigger occurs the un-integrated data is cumulated and the sum of all seconds of every unincorporated data
    point is added up.

    `incorporation_delay_per_training_second` servers as a conversion rate between budget (training time) and regret
    metric (incorporation latency).

    When the regret metric exceeds the training-time based budget metric, a trigger is fired.
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


class AvoidableMisclassificationCostTriggerConfig(_CostTriggerConfig, _InternalPerformanceTriggerConfig):
    """Cost aware trigger policy configuration that using the number of
    avoidable misclassifications integration latency as a regret metric.

    `avoidable_misclassification_latency_per_training_second` servers as a conversion rate between budget (training time)
    and regret metric (misclassifications).

    When a the regret metric exceeds the budget, a trigger is fired.

    This policy is an extension of performance aware triggers.

    Offers the same set of `decision_criteria` as `PerformanceTriggerConfig`
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

    @property
    def conversion_factor(self) -> float:
        """Conversion factor between budget (training time) and regret metric
        (avoidable misclassifications).

        Unifies names for easier downstream usage.
        """
        return self.avoidable_misclassification_latency_per_training_second


CostTriggerConfig = Annotated[
    AvoidableMisclassificationCostTriggerConfig | DataIncorporationLatencyCostTriggerConfig,
    Field(discriminator="id"),
]
