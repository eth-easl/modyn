from typing import Annotated, Literal, Union

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class HypothesisTestDecisionCriterion(ModynBaseModel):
    id: Literal["HypothesisTestDecisionCriterion"] = "HypothesisTestDecisionCriterion"

    @property
    def needs_calibration(self) -> bool:
        return False


class ThresholdDecisionCriterion(ModynBaseModel):
    id: Literal["ThresholdDecisionCriterion"] = "ThresholdDecisionCriterion"
    threshold: float

    @property
    def needs_calibration(self) -> bool:
        return False


class DynamicThresholdCriterion(ModynBaseModel):
    id: Literal["DynamicThresholdCriterion"] = "DynamicThresholdCriterion"
    window_size: int
    percentile_threshold: float = Field(
        0.05, description="The percentile that a threshold has to be in to trigger a drift event."
    )

    @property
    def needs_calibration(self) -> bool:
        return True


DecisionCriterion = Annotated[
    Union[
        HypothesisTestDecisionCriterion,
        ThresholdDecisionCriterion,
        DynamicThresholdCriterion,
    ],
    Field(discriminator="id"),
]


class BaseMetric(ModynBaseModel):
    decision_criterion: DecisionCriterion
