from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel

# -------------------------------------------------------------------------------------------------------------------- #
#                                                  Ensemble Strategies                                                 #
# -------------------------------------------------------------------------------------------------------------------- #


class BaseEnsembleStrategy(ModynBaseModel):
    @property
    def aggregate_decision_func(self) -> Callable[[dict[str, bool]], bool]:
        """Returns:
        Function that aggregates the decisions of the individual triggers."""
        raise NotImplementedError


class MajorityVoteEnsembleStrategy(BaseEnsembleStrategy):
    id: Literal["MajorityVote"] = Field("MajorityVote")

    @property
    def aggregate_decision_func(self) -> Callable[[dict[str, bool]], bool]:
        return lambda decisions: sum(decisions.values()) > len(decisions) / 2


class AtLeastNEnsembleStrategy(BaseEnsembleStrategy):
    id: Literal["AtLeastN"] = Field("AtLeastN")

    n: int = Field(
        description="The minimum number of triggers that need to trigger for the ensemble to trigger.",
        ge=1,
    )

    @property
    def aggregate_decision_func(self) -> Callable[[dict[str, bool]], bool]:
        return lambda decisions: sum(decisions.values()) >= self.n


class CustomEnsembleStrategy(BaseEnsembleStrategy):
    id: Literal["Custom"] = Field("Custom")

    aggregation_function: Callable[[dict[str, bool]], bool] = Field(
        description="The function that aggregates the decisions of the individual triggers."
    )

    @property
    def aggregate_decision_func(self) -> Callable[[dict[str, bool]], bool]:
        return self.aggregation_function


EnsembleStrategy = Annotated[
    MajorityVoteEnsembleStrategy | AtLeastNEnsembleStrategy | CustomEnsembleStrategy,
    Field(discriminator="id"),
]

# -------------------------------------------------------------------------------------------------------------------- #
#                                                    EnsembleTrigger                                                   #
# -------------------------------------------------------------------------------------------------------------------- #


class EnsembleTriggerConfig(ModynBaseModel):
    id: Literal["EnsembleTrigger"] = Field("EnsembleTrigger")

    subtriggers: dict[str, Any] = Field(
        default_factory=dict,
        description="The sub-triggers keyed by distinct references that will be consulted for the ensemble trigger.",
    )

    ensemble_strategy: EnsembleStrategy = Field(
        description="The strategy that will be used to aggregate the decisions of the individual triggers."
    )
