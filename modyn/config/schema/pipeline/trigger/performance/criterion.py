from typing import Annotated, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel

# -------------------------------------------------------------------------------------------------------------------- #
#                                                 PerformanceCriterion                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
# uses the evaluation results to derive a triggering decision


class _PerformanceThresholdCriterion(ModynBaseModel):
    metric: str = Field(
        description="The metric that should be used for the comparison. Name as defined in the evaluation config."
    )


class StaticPerformanceThresholdCriterion(_PerformanceThresholdCriterion):
    id: Literal["StaticPerformanceThresholdCriterion"] = Field("StaticPerformanceThresholdCriterion")
    metric_threshold: float = Field(
        0.0,
        description=(
            "The minimum target metric value that the model should achieve. If the performance is NOT reached "
            "a trigger will be forced."
        ),
    )
    needs_calibration: Literal[False] = Field(False)


class _DynamicPerformanceThresholdCriterion(_PerformanceThresholdCriterion):
    """Triggers after comparison of current performance with the a rolling
    average of historic performances after triggers."""

    window_size: int = Field(10)
    needs_calibration: Literal[True] = Field(True)


class DynamicQuantilePerformanceThresholdCriterion(_DynamicPerformanceThresholdCriterion):
    """Dynamic threshold based on a extremeness quantile of the previous
    distance values."""

    id: Literal["DynamicQuantilePerformanceThresholdCriterion"] = Field("DynamicQuantilePerformanceThresholdCriterion")
    quantile: float = Field(
        0.05,
        description=(
            "The quantile that a threshold has to trigger. "
            "0.05 will only trigger in the most extreme 5% of cases. Hence the triggering "
            "threshold is more extreme than 95% of the previous values."
        ),
    )


class DynamicRollingAveragePerformanceThresholdCriterion(_DynamicPerformanceThresholdCriterion):
    """Triggers when a new distance value deviates from the rolling average by
    a certain amount or percentage."""

    id: Literal["DynamicRollingAveragePerformanceThresholdCriterion"] = Field(
        "DynamicRollingAveragePerformanceThresholdCriterion"
    )
    deviation: float = Field(
        0.05,
        description="The deviation from the rolling average that triggers a drift event.",
    )
    absolute: bool = Field(
        False,
        description="Whether the deviation is absolute or relative to the rolling average.",
    )


# -------------------------------------------------------------------------------------------------------------------- #
#                                           NumberMisclassificationCriterion                                           #
# -------------------------------------------------------------------------------------------------------------------- #


class _NumberAvoidableMisclassificationCriterion(ModynBaseModel):
    """Trigger based on the cumulated number of avoidable misclassifications.

    An avoidable misclassification is a misclassification that would have been avoided if a trigger would have been.
    E.g. if we currency see an accuracy of 80% but expect 95% with a trigger, 15% of the samples are avoidable
    misclassifications.

    We estimate the number of avoidable misclassifications with the measured and expected performance.

    The cumulated number of misclassifications can be seen a regret of not triggering.

    Advantage: cumulative metric allows to account for persisting slightly degraded performance. Eventually
    the cumulated number of misclassifications will trigger a trigger.
    """

    expected_accuracy: float | None = Field(
        None,
        description=(
            "The expected accuracy of the model. Used to estimate the number of avoidable misclassifications. "
            "If not set, the expected performance will be inferred dynamically with a rolling average."
        ),
    )
    allow_reduction: bool = Field(
        False,
        description=(
            "If True, the cumulated number of misclassifications will be lowered through evaluations that are "
            "better than the expected performance."
        ),
    )
    needs_calibration: Literal[True] = Field(True)


class StaticNumberAvoidableMisclassificationCriterion(_NumberAvoidableMisclassificationCriterion):
    id: Literal["StaticNumberMisclassificationCriterion"] = Field("StaticNumberMisclassificationCriterion")
    avoidable_misclassification_threshold: int = Field(
        description="The threshold for the misclassification rate that will invoke a trigger."
    )


# -------------------------------------------------------------------------------------------------------------------- #
#                                                         Union                                                        #
# -------------------------------------------------------------------------------------------------------------------- #

PerformanceTriggerCriterion = Annotated[
    StaticPerformanceThresholdCriterion
    | DynamicQuantilePerformanceThresholdCriterion
    | DynamicRollingAveragePerformanceThresholdCriterion
    | StaticNumberAvoidableMisclassificationCriterion,
    Field(discriminator="id"),
]
