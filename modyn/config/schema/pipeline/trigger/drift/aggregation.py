"""Support the aggregation of the decisions of multiple drift metrics."""

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel

from .result import MetricResult


class MajorityVoteDriftAggregationStrategy(ModynBaseModel):
    id: Literal["MajorityVote"] = Field("MajorityVote")

    @property
    def aggregate_decision_func(self) -> Callable[[dict[str, MetricResult]], bool]:
        return lambda decisions: sum(decision.is_drift for decision in decisions.values()) > len(decisions) / 2


class AtLeastNDriftAggregationStrategy(ModynBaseModel):
    id: Literal["AtLeastN"] = Field("AtLeastN")

    n: int = Field(description="The minimum number of triggers that need to trigger for the ensemble to trigger.", ge=1)

    @property
    def aggregate_decision_func(self) -> Callable[[dict[str, MetricResult]], bool]:
        return lambda decisions: sum(decision.is_drift for decision in decisions.values()) >= self.n


DriftAggregationStrategy = Annotated[
    MajorityVoteDriftAggregationStrategy | AtLeastNDriftAggregationStrategy,
    Field(discriminator="id"),
]
