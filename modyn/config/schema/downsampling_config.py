from typing import Annotated, Literal, Union

from modyn.config.schema.modyn_base_model import ModynBaseModel
from pydantic import Field


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

    strategy: Literal["Craig"]
    score_metric: Literal["LeastConfidence", "Entropy", "Margin"] = Field(
        description="the metric used to score uncertainty for the datapoints"
    )
    balance: bool = Field(False, description="If True, the samples are balanced.")


class KcenterGreedyDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the KcenterGreedy downsampling strategy."""

    strategy: Literal["KcenterGreedy"]
    balance: bool = Field(False, description="If True, the samples are balanced.")


class GradMatchDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the GradMatch downsampling strategy."""

    strategy: Literal["GradMatch"]
    balance: bool = Field(False, description="If True, the samples are balanced.")


class CraigDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the Craig downsampling strategy."""

    strategy: Literal["Craig"]
    selection_batch: int = Field(64, description="The batch size for the selection.")
    balance: bool = Field(False, description="If True, the samples are balanced.")
    greedy: Literal["NaiveGreedy", "LazyGreedy", "StochasticGreedy", "ApproximateLazyGreedy"] = Field(
        "NaiveGreedy", description="The greedy strategy to use."
    )


class LossDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the Loss downsampling strategy."""

    strategy: Literal["Loss"]


class SubmodularDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the Submodular downsampling strategy."""

    strategy: Literal["Submodular"]
    submodular_function: Literal["FacilityLocation", "GraphCut", "LogDeterminant"]
    submodular_optimizer: Literal["NaiveGreedy", "LazyGreedy", "StochasticGreedy", "ApproximateLazyGreedy"] = Field(
        "NaiveGreedy", description="The greedy strategy to use."
    )
    selection_batch: int = Field(64, description="The batch size for the selection.")
    balance: bool = Field(False, description="If True, the samples are balanced.")


class GradNormDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the GradNorm downsampling strategy."""

    strategy: Literal["GradNorm"]


class NoDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the No downsampling strategy."""

    strategy: Literal["No"]


class RHOLossDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the RHO Loss downsampling strategy."""

    strategy: Literal["RHOLoss"]


DownsamplingConfig = Annotated[
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
