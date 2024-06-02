from __future__ import annotations

from typing import Literal, Dict, Any, List
from typing_extensions import Self

from pydantic import Field, model_validator

from modyn.config.schema.modyn_base_model import ModynBaseModel

OptimizerSource = Literal["PyTorch", "APEX"]


class OptimizerParamGroup(ModynBaseModel):
    """Configuration for a parameter group."""

    module: str = Field(description="A set of parameters.")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional configuration for the parameter group. (e.g. learning rate)",
    )


class OptimizerConfig(ModynBaseModel):
    """Configuration for the optimizer used during training."""

    name: str = Field(description="The name of the optimizer (like an ID).")
    algorithm: str = Field(description="The type of the optimizer (e.g. SGD).")
    source: OptimizerSource = Field(description="The framework/package the optimizer comes from.")
    param_groups: List[OptimizerParamGroup] = Field(
        description="An array of the parameter groups (parameters and optional configs) this optimizer supervises.",
        min_length=1,
    )


class OptimizationCriterion(ModynBaseModel):
    """Configuration for the optimization criterion that we optimize."""

    name: str = Field(description="The name of the criterion that the pipeline uses (e.g., CrossEntropyLoss).")
    config: Dict[str, Any] = Field(default_factory=dict, description="Optional configuration of the criterion.")


LrSchedulerSource = Literal["PyTorch", "Custom"]


class LrSchedulerConfig(ModynBaseModel):
    """Configuration for the Torch-based Learning Rate (LR) scheduler used for training."""

    name: str = Field(description="The name of the LR scheduler.")
    source: LrSchedulerSource = Field(description="Source of the LR scheduler.")
    optimizers: List[str] = Field(
        description="List of optimizers that this scheduler is responsible for.",
        min_length=1,
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional configuration of the lr scheduler. Passed to the lr scheduler as a dict.",
    )

    @model_validator(mode="after")
    def validate_optimizers(self) -> Self:
        if self.source == "PyTorch" and len(self.optimizers) != 1:
            raise ValueError("In case a PyTorch LR scheduler is used, the optimizers list should have only one item.")
        return self
