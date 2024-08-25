from __future__ import annotations

from typing import Annotated, Any, ForwardRef, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel
from modyn.config.schema.pipeline.trigger.drift.detection_window import (
    AmountWindowingStrategy,
    DriftWindowingStrategy,
)
from modyn.config.schema.pipeline.trigger.simple import SimpleTriggerConfig

from .aggregation import DriftAggregationStrategy, MajorityVoteDriftAggregationStrategy
from .alibi_detect import AlibiDetectDriftMetric
from .evidently import EvidentlyDriftMetric

__TriggerConfig = ForwardRef("TriggerConfig", is_class=True)

DriftMetric = Annotated[
    EvidentlyDriftMetric | AlibiDetectDriftMetric,
    Field(discriminator="id"),
]


class DataDriftTriggerConfig(ModynBaseModel):
    id: Literal["DataDriftTrigger"] = Field("DataDriftTrigger")

    reset_current_window_on_trigger: Any = None
    detection_interval: __TriggerConfig | None = Field(  # type: ignore[valid-type]
        None,
        description="The Trigger policy to determine the interval at which drift detection is performed.",
    )  # currently not used
    detection_interval_data_points: int = Field(
        1000,
        description="The number of samples in the interval after which drift detection is performed.",
        ge=1,
    )

    windowing_strategy: DriftWindowingStrategy = Field(
        AmountWindowingStrategy(),
        description="Which windowing strategy to use for current and reference data",
    )

    warmup_intervals: int | None = Field(
        None,
        description=(
            "The number of intervals before starting to use the drift detection. Some "
            "`DecisionCriteria` use this to calibrate the threshold. During the warmup, a simpler `warmup_policy` "
            "is consulted for the triggering decision."
        ),
    )
    warmup_policy: SimpleTriggerConfig | None = Field(
        None,
        description=(
            "The policy to use for triggering during the warmup phase of the drift policy. "
            "Metrics that don't need calibration can ignore this."
        ),
    )

    metrics: dict[str, DriftMetric] = Field(
        min_length=1,
        description="The metrics used for drift detection keyed by a reference.",
    )
    aggregation_strategy: DriftAggregationStrategy = Field(
        MajorityVoteDriftAggregationStrategy(),
        description="The strategy to aggregate the decisions of the individual drift metrics.",
    )

    # @model_validator(mode="after")
    # def warmup_policy_requirement(self) -> Self:
    #     """Assert whether the warmup policy is set when a metric needs
    #     calibration."""
    #     for metric in self.metrics.values():
    #         if (
    #             metric.decision_criterion.needs_calibration
    #             and self.warmup_policy is None
    #         ):
    #             raise ValueError(
    #                 "A warmup policy is required for metrics that need calibration."
    #             )
    #     return self
