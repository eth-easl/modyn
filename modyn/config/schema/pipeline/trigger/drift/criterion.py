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


class DynamicPercentileThresholdCriterion(_DynamicThresholdCriterion):
    """Dynamic threshold based on a extremeness percentile of the previous
    distance values."""

    id: Literal["DynamicPercentileThresholdCriterion"] = "DynamicPercentileThresholdCriterion"
    percentile: float = Field(
        0.05,
        description="The percentile that a threshold has to be in to trigger a drift event.",
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


DynamicThresholdCriterion = DynamicPercentileThresholdCriterion | DynamicRollingAverageThresholdCriterion

DriftDecisionCriterion = Annotated[
    ThresholdDecisionCriterion | DynamicPercentileThresholdCriterion | DynamicRollingAverageThresholdCriterion,
    Field(discriminator="id"),
]
