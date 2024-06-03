from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from modyn.config.schema.data.data_config import DataConfig
from modyn.config.schema.modyn_base_model import ModynBaseModel
from modyn.config.schema.optimizer.optimizer_config import LrSchedulerConfig, OptimizationCriterion, OptimizerConfig
from pydantic import Field, model_validator
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


class ILTrainingConfig(ModynBaseModel):
    num_workers: int = Field("Number of workers to use for training.", min=1)
    il_model_id: str = Field("The model class name to use as the IL model.")
    il_model_config: dict = Field(
        default_factory=dict, description="Configuration dictionary that will be passed to the model on initialization."
    )
    amp: bool = Field(False, description="Whether to use automatic mixed precision.")
    device: str = Field(description="The device to use to train IL model.")
    batch_size: int = Field(description="The batch size to use for training the IL model.", min=1)
    epochs: int = Field(description="The number of epochs to train the IL model.", min=1)
    optimizers: List[OptimizerConfig] = Field(description="The optimizer configuration for the IL model.")
    optimization_criterion: OptimizationCriterion = Field(
        description="Configuration for the optimization criterion that we optimize",
    )
    lr_scheduler: Optional[LrSchedulerConfig] = Field(
        None,
        description="Configuration for the Torch-based Learning Rate (LR) scheduler used for training.",
    )
    grad_scaler_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration for the torch.cuda.amp.GradScaler. Effective only when amp is enabled.",
    )



class RHOLossDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the RHO Loss downsampling strategy."""

    strategy: Literal["RHOLoss"] = "RHOLoss"
    sample_then_batch: Literal[False] = False
    holdout_set_ratio: int = Field(
        description=("How much of the training set is used as the holdout set."),
        min=0,
        max=100,
    )
    il_training_config: ILTrainingConfig = Field(description="The configuration for the IL training.")


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
