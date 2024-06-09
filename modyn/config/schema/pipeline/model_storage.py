from typing import Any, Literal, Optional, Union

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class _BaseModelStrategy(ModynBaseModel):
    """Base class for the model storage strategies."""

    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration dictionary that will be passed to the strategy.",
    )
    zip: Optional[bool] = Field(None, description="Whether to zip the file in the end.")
    zip_algorithm: str = Field(
        default="ZIP_DEFLATED",
        description="Which zip algorithm to use. Default is ZIP_DEFLATED.",
    )


class FullModelStrategy(_BaseModelStrategy):
    """Which full model strategy is used for the model storage."""

    name: Literal["PyTorchFullModel", "BinaryFullModel"] = Field(description="Name of the full model strategy.")


class IncrementalModelStrategy(_BaseModelStrategy):
    """Which incremental model strategy is used for the model storage."""

    name: Literal["WeightsDifference"] = Field(
        description="Name of the incremental model strategy. We currently support WeightsDifference."
    )
    full_model_interval: Optional[int] = Field(
        None, description="In which interval we are using the full model strategy."
    )


ModelStrategy = Union[FullModelStrategy, IncrementalModelStrategy]


class PipelineModelStorageConfig(ModynBaseModel):
    full_model_strategy: FullModelStrategy = Field(description="Which full model strategy is used.")
    incremental_model_strategy: Optional[IncrementalModelStrategy] = Field(
        None, description="Which incremental model strategy is used."
    )
