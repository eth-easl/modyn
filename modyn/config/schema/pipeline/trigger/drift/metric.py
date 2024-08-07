from typing import Annotated, Literal, Union

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel


class HypothesisTestDecisionCriterion(ModynBaseModel):
    id: Literal["HypothesisTestDecisionCriterion"] = "HypothesisTestDecisionCriterion"
    needs_calibration: Literal[False] = Field(False)


class ThresholdDecisionCriterion(ModynBaseModel):
    id: Literal["ThresholdDecisionCriterion"] = "ThresholdDecisionCriterion"
    threshold: float
    needs_calibration: Literal[False] = Field(False)


class DynamicThresholdCriterion(ModynBaseModel):
    id: Literal["DynamicThresholdCriterion"] = "DynamicThresholdCriterion"
    window_size: int
    percentile_threshold: float = Field(
        0.05,
        description="The percentile that a threshold has to be in to trigger a drift event.",
    )
    needs_calibration: Literal[True] = Field(True)


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
