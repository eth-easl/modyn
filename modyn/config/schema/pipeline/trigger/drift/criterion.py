from typing import Annotated, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel


class ThresholdDecisionCriterion(ModynBaseModel):
    id: Literal["ThresholdDecisionCriterion"] = "ThresholdDecisionCriterion"
    threshold: float
    needs_calibration: Literal[False] = Field(False)


class DynamicThresholdCriterion(ModynBaseModel):
    id: Literal["DynamicThresholdCriterion"] = "DynamicThresholdCriterion"
    window_size: int = Field(10)
    percentile: float = Field(
        0.05,
        description="The percentile that a threshold has to be in to trigger a drift event.",
    )
    needs_calibration: Literal[True] = Field(True)


DriftDecisionCriterion = Annotated[
    ThresholdDecisionCriterion | DynamicThresholdCriterion,
    Field(discriminator="id"),
]
