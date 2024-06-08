from __future__ import annotations

from typing import List

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field, model_validator

from .strategy import EvalStrategyConfig, UntilNextTriggerEvalStrategyConfig
from .trigger import AfterTrainingEvalTriggerConfig, EvalTriggerConfig


class EvalHandlerConfig(ModynBaseModel):
    name: str = Field(
        "Modyn EvalHandler", description="The name of the evaluation handler used to identify its outputs in the log"
    )
    strategy: EvalStrategyConfig = Field(description="Defining the strategy and data range to be evaluated on.")
    trigger: EvalTriggerConfig = Field(None, description="Configures when the evaluation should be performed.")
    datasets: List[str] = Field(
        description="All datasets on which the model is evaluated.",
        min_length=1,
    )
    """ Note: the datasets have to be defined in the root EvaluationConfig model."""

    @model_validator(mode="after")
    def check_trigger(self) -> "EvalHandlerConfig":
        if isinstance(self.strategy, UntilNextTriggerEvalStrategyConfig) and not (
            isinstance(self.trigger, AfterTrainingEvalTriggerConfig)
        ):
            raise ValueError("UntilNextTriggerEvalStrategyConfig can only be used with AfterTrainingEvalTriggerConfig.")
        return self
