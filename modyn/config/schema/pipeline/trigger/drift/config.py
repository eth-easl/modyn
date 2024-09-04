from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from modyn.config.schema.pipeline.trigger.common.batched import BatchedTriggerConfig
from modyn.config.schema.pipeline.trigger.drift.detection_window import (
    AmountWindowingStrategy,
    DriftWindowingStrategy,
)

from .aggregation import DriftAggregationStrategy, MajorityVoteDriftAggregationStrategy
from .alibi_detect import AlibiDetectDriftMetric
from .evidently import EvidentlyDriftMetric

DriftMetric = Annotated[
    EvidentlyDriftMetric | AlibiDetectDriftMetric,
    Field(discriminator="id"),
]


class DataDriftTriggerConfig(BatchedTriggerConfig):
    id: Literal["DataDriftTrigger"] = Field("DataDriftTrigger")

    windowing_strategy: DriftWindowingStrategy = Field(
        AmountWindowingStrategy(),
        description="Which windowing strategy to use for current and reference data",
    )

    sample_size: int | None = Field(
        5000,
        description=(
            "The number of samples to use for drift detection. If the windows are bigger than this, "
            "samples are randomly drawn from the window. None does not limit the number of samples."
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

    @model_validator(mode="after")
    def warmup_policy_requirement(self) -> Self:
        """Assert whether the warmup policy is set when a metric needs
        calibration."""
        for metric in self.metrics.values():
            if metric.decision_criterion.needs_calibration and self.warmup_policy is None:
                raise ValueError("A warmup policy is required for drift policies that need calibration.")
        return self
