from typing import Annotated, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel

# -------------------------------------------------------------------------------------------------------------------- #
#                                                 PerformanceCriterion                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
# uses the evaluation results directory to derive a triggering decision


class _NamedMetricPerformanceCriterion(ModynBaseModel):
    metric: str = Field(
        description="The metric that should be used for the comparison. Name as defined in the evaluation config."
    )


class StaticPerformanceThresholdCriterion(_NamedMetricPerformanceCriterion):
    id: Literal["StaticPerformanceThresholdCriterion"] = Field("StaticPerformanceThresholdCriterion")
    metric_threshold: float = Field(
        0.0,
        description=(
            "The minimum target metric value that the model should achieve. If the performance is NOT reached "
            "a trigger will be forced."
        ),
    )


class DynamicPerformanceThresholdCriterion(_NamedMetricPerformanceCriterion):
    """Triggers after comparison of current performance with the a rolling
    average of historic performances after triggers."""

    id: Literal["DynamicPerformanceThresholdCriterion"] = Field("DynamicPerformanceThresholdCriterion")
    allowed_deviation: float = Field(
        0.05,
        description=(
            "The allowed deviation from the expected performance. Will only trigger if the performance is "
            "below the expected performance minus the allowed deviation."
        ),
    )


# TODO: drift: rolling average + x % (instead of percentile)


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


class StaticNumberAvoidableMisclassificationCriterion(_NumberAvoidableMisclassificationCriterion):
    id: Literal["StaticNumberMisclassificationCriterion"] = Field("StaticNumberMisclassificationCriterion")
    avoidable_misclassification_threshold: int = Field(
        description="The threshold for the misclassification rate that will invoke a trigger."
    )
    allow_reduction: bool = Field(
        False,
        description=(
            "If True, the cumulated number of misclassifications will be lowered through evaluations that are "
            "better than the expected performance."
        ),
    )


# -------------------------------------------------------------------------------------------------------------------- #
#                                                         Union                                                        #
# -------------------------------------------------------------------------------------------------------------------- #

PerformanceTriggerCriterion = Annotated[
    StaticPerformanceThresholdCriterion
    | DynamicPerformanceThresholdCriterion
    | StaticNumberAvoidableMisclassificationCriterion,
    Field(discriminator="id"),
]
