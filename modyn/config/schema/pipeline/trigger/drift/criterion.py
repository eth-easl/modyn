from typing import Annotated, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel


class ThresholdDecisionCriterion(ModynBaseModel):
    id: Literal["ThresholdDecisionCriterion"] = "ThresholdDecisionCriterion"
    threshold: float
    needs_calibration: Literal[False] = Field(False)


class _DynamicThresholdCriterion(ModynBaseModel):
    window_size: int = Field(10)
    needs_calibration: Literal[True] = Field(True)


class DynamicQuantileThresholdCriterion(_DynamicThresholdCriterion):
    """Dynamic threshold based on a extremeness quantile of the previous
    distance values."""

    id: Literal["DynamicQuantileThresholdCriterion"] = "DynamicQuantileThresholdCriterion"
    quantile: float = Field(
        0.05,
        description=(
            "The quantile that a threshold has to be in to trigger a drift event. "
            "0.05 will only trigger in the most extreme 5% of cases. Hence the triggering "
            "threshold is more extreme than 95% of the previous values."
        ),
        min=0.0,
        max=1.0,
    )


class DynamicRollingAverageThresholdCriterion(_DynamicThresholdCriterion):
    """Triggers when a new distance value deviates from the rolling average by
    a certain amount or percentage."""

    id: Literal["DynamicRollingAverageThresholdCriterion"] = "DynamicRollingAverageThresholdCriterion"
    deviation: float = Field(
        0.05,
        description="The deviation from the rolling average that triggers a drift event.",
    )
    absolute: bool = Field(
        False,
        description="Whether the deviation is absolute or relative to the rolling average.",
    )


DynamicThresholdCriterion = DynamicQuantileThresholdCriterion | DynamicRollingAverageThresholdCriterion

DriftDecisionCriterion = Annotated[
    ThresholdDecisionCriterion | DynamicQuantileThresholdCriterion | DynamicRollingAverageThresholdCriterion,
    Field(discriminator="id"),
]
