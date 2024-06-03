from __future__ import annotations

from typing import Annotated, List, Literal, Union

from modyn.config.schema.modyn_base_model import ModynBaseModel
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self


class BaseDownsamplingConfig(ModynBaseModel):
    """Config for the downsampling strategy of SelectionStrategy."""

    sample_then_batch: bool = Field(
        False,
        description=(
            "If True, the samples are first sampled and then batched and supplied to the training loop. If False, "
            "the datapoints are first divided into batches and then sampled."
        ),
    )
    ratio: int = Field(
        description="Ratio post_sampling_size/pre_sampling_size. E.g. with 160 records and a ratio of 50 we keep 80.",
        min=0,
        max=100,
    )
    period: int = Field(
        1,
        description=(
            "In multi-epoch training and sample_then_batch, how frequently the data is selected. "
            "`1` selects every epoch. To select once per trigger, set this parameter to 0."
            "When sample_then_batch is False, this parameter is ignored."
        ),
        min=0,
    )


class UncertaintyDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the Craig downsampling strategy."""

    strategy: Literal["Uncertainty"] = "Uncertainty"
    score_metric: Literal["LeastConfidence", "Entropy", "Margin"] = Field(
        description="the metric used to score uncertainty for the datapoints"
    )
    balance: bool = Field(False, description="If True, the samples are balanced.")


class KcenterGreedyDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the KcenterGreedy downsampling strategy."""

    strategy: Literal["KcenterGreedy"] = "KcenterGreedy"
    balance: bool = Field(False, description="If True, the samples are balanced.")


class GradMatchDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the GradMatch downsampling strategy."""

    strategy: Literal["GradMatch"] = "GradMatch"
    balance: bool = Field(False, description="If True, the samples are balanced.")


class CraigDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the Craig downsampling strategy."""

    strategy: Literal["Craig"] = "Craig"
    selection_batch: int = Field(64, description="The batch size for the selection.")
    balance: bool = Field(False, description="If True, the samples are balanced.")
    greedy: Literal["NaiveGreedy", "LazyGreedy", "StochasticGreedy", "ApproximateLazyGreedy"] = Field(
        "NaiveGreedy", description="The greedy strategy to use."
    )


class LossDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the Loss downsampling strategy."""

    strategy: Literal["Loss"] = "Loss"


class SubmodularDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the Submodular downsampling strategy."""

    strategy: Literal["Submodular"] = "Submodular"
    submodular_function: Literal["FacilityLocation", "GraphCut", "LogDeterminant"]
    submodular_optimizer: Literal["NaiveGreedy", "LazyGreedy", "StochasticGreedy", "ApproximateLazyGreedy"] = Field(
        "NaiveGreedy", description="The greedy strategy to use."
    )
    selection_batch: int = Field(64, description="The batch size for the selection.")
    balance: bool = Field(False, description="If True, the samples are balanced.")


class GradNormDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the GradNorm downsampling strategy."""

    strategy: Literal["GradNorm"] = "GradNorm"


class NoDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the No downsampling strategy."""

    strategy: Literal["No"] = "No"
    ratio: Literal[100] = 100


class RHOLossDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the RHO Loss downsampling strategy."""

    strategy: Literal["RHOLoss"] = "RHOLoss"
    sample_then_batch: Literal[False] = False
    holdout_set_ratio: int = Field(
        description=("How much of the training set is used as the holdout set."),
        min=0,
        max=100,
    )


class RS2DownsamplingConfig(BaseDownsamplingConfig):
    """Config for the RS2 downsampling strategy."""

    strategy: Literal["RS2"] = "RS2"
    with_replacement: bool = Field(
        description="Whether we resample from the full TTS each epoch (= True) or train on all the data with a different subset each epoch (= False)."
    )

    @field_validator("sample_then_batch")
    def sample_then_batch_must_be_true(cls, v):
        if not v:
            raise ValueError("sample_then_batch must be set to True for this config.")
        return v

    @field_validator("period")
    def only_support_period_one(cls, v):
        if v != 0:
            # RS2 requires us to resample every epoch.
            raise ValueError("period must be set to 1 for this config.")
        return v


SingleDownsamplingConfig = Annotated[
    Union[
        UncertaintyDownsamplingConfig,
        KcenterGreedyDownsamplingConfig,
        GradMatchDownsamplingConfig,
        CraigDownsamplingConfig,
        LossDownsamplingConfig,
        SubmodularDownsamplingConfig,
        GradNormDownsamplingConfig,
        NoDownsamplingConfig,
        RHOLossDownsamplingConfig,
    ],
    Field(discriminator="strategy"),
]


class MultiDownsamplingConfig(ModynBaseModel):
    downsampling_list: List[SingleDownsamplingConfig] = Field(description="An array of downsampling strategies.")
    downsampling_thresholds: List[int] = Field(
        description=(
            "A list of thresholds to switch from a downsampler to another. The i-th threshold is used for the "
            "transition from the i-th downsampler to the (i+1)-th. This array should have one less item than the list "
            "of downsamplers. For example, if we have 3 downsamplers [A, B, C], and two thresholds [5, 10], the "
            "downsampler A is used for triggers 0-4, B for triggers 5-9, and C for triggers 10 and above."
        )
    )

    @model_validator(mode="after")
    def validate_downsampling_thresholds(self) -> Self:
        if len(self.downsampling_thresholds) != len(self.downsampling_list) - 1:
            raise ValueError("The downsampling_thresholds list should have one less item than the downsampling_list.")
        return self
