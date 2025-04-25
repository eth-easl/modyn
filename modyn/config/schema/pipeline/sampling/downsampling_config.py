from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from modyn.config.schema.base_model import ModynBaseModel

from ..training import CheckpointingConfig, TrainingConfig


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
        description=(
            "Ratio post_sampling_size/pre_sampling_size * ratio_max. "
            "For the default of ratio_max of 100, this implies percent, "
            "e.g., with 160 records and a ratio of 50 we keep 80."
        ),
        min=0,
    )
    ratio_max: int = Field(
        description=(
            "Reference maximum ratio value. Defaults to 100, which implies percent."
            " If you set this to 1000, ratio describes promille instead."
        ),
        default=100,
        min=1,
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

    @model_validator(mode="after")
    def validate_ratio(self) -> Self:
        if self.ratio > self.ratio_max:
            raise ValueError("ratio cannot be greater than ratio_max.")
        return self


# These are options to approximate the full gradients used in several selection strategies.
# LastLayer: The full gradient is approximated by the gradient of the last layer.
# LastLayerWithEmbedding: The full gradient is approximated by the gradients of the last layer and the embedding layer.
# They are concatenated and used to represent the full gradient.
FullGradApproximation = Literal["LastLayer", "LastLayerWithEmbedding"]


class UncertaintyDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the Craig downsampling strategy."""

    strategy: Literal["Uncertainty"] = "Uncertainty"
    score_metric: Literal["LeastConfidence", "Entropy", "Margin"] = Field(
        description="the metric used to score uncertainty for the datapoints"
    )
    balance: bool = Field(False, description="If True, the samples are downsambalanced.")

class PerTokenUncertaintyDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the Uncertainty downsampling strategy."""

    strategy: Literal["TokenUncertainty"] = "TokenUncertainty"
    generative: bool = Field(
        True,
        description="Whether this strategy applies to generative tasks (must be True)."
    )
    score_metric: Literal["LeastConfidence", "Entropy", "Margin"] = Field(
        description="Metric used to score token uncertainty."
    )
    # Mutually exclusive per-sample selection options
    per_sample_top_k: int | None = Field(
        None,
        description="If set, select top-K tokens per sample by uncertainty."
    )
    per_sample_ratio: float | None = Field(
        None,
        description="If set, select this fraction of tokens per sample (per_sample_ratio * length / ratio_max)."
    )
    balance: bool = Field(
        False,
        description="If True, selected tokens are balanced across classes or buckets."
    )
    weight_per_sample: bool = Field(
        False,
        description=(
            "If True, perform sample-level selection (all tokens in a selected sample get weight=1, others 0). "
            "If False, perform token-level selection."
        ),
    )

    @model_validator(mode="after")
    def validate_per_sample(cls, values: Self) -> Self:
        if values.per_sample_top_k is not None and values.per_sample_ratio is not None:
            raise ValueError("Specify only one of 'per_sample_top_k' or 'per_sample_ratio'.")
        return values

class KcenterGreedyDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the KcenterGreedy downsampling strategy."""

    strategy: Literal["KcenterGreedy"] = "KcenterGreedy"
    balance: bool = Field(False, description="If True, the samples are balanced.")


class GradMatchDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the GradMatch downsampling strategy."""

    strategy: Literal["GradMatch"] = "GradMatch"
    balance: bool = Field(False, description="If True, the samples are balanced.")
    full_grad_approximation: FullGradApproximation = Field(default="LastLayer")


class CraigDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the Craig downsampling strategy."""

    strategy: Literal["Craig"] = "Craig"
    selection_batch: int = Field(64, description="The batch size for the selection.")
    balance: bool = Field(False, description="If True, the samples are balanced.")
    greedy: Literal["NaiveGreedy", "LazyGreedy", "StochasticGreedy", "ApproximateLazyGreedy"] = Field(
        "NaiveGreedy", description="The greedy strategy to use."
    )
    full_grad_approximation: FullGradApproximation = Field(default="LastLayer")


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
    full_grad_approximation: FullGradApproximation = Field(default="LastLayer")


class GradNormDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the GradNorm downsampling strategy."""

    strategy: Literal["GradNorm"] = "GradNorm"


class NoDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the No downsampling strategy."""

    strategy: Literal["No"] = "No"
    ratio: Literal[100] = 100


class NoCheckpointingConfig(CheckpointingConfig):
    activated: Literal[False] = False
    interval: Literal[None] = None
    path: Literal[None] = None


class ILTrainingConfig(TrainingConfig):
    # new fields introduced
    il_model_id: str = Field(description="The model class name to use as the IL model.")
    il_model_config: dict = Field(
        default_factory=dict, description="Configuration dictionary that will be passed to the model on initialization."
    )

    # hardcode values that are not relevant in the IL training
    gpus: Literal[1] = 1
    num_samples_to_pass: Literal[None] = None
    initial_model: Literal["random"] = "random"
    initial_model_id: Literal[None] = None
    checkpointing: NoCheckpointingConfig = Field(default=NoCheckpointingConfig())
    drop_last_batch: bool = Field(
        default=False, description="Whether to drop the last batch if it is smaller than the batch size."
    )


class RHOLossDownsamplingConfig(BaseDownsamplingConfig):
    """Config for the RHO Loss downsampling strategy."""

    strategy: Literal["RHOLoss"] = "RHOLoss"
    holdout_set_strategy: Literal["Simple", "Twin"] = Field(
        description="Simple: holdout set is a subset randomly sampled from the training set based on the"
        "holdout_set_ratio. The holdout set is used to train the il model and the original training set is"
        "used to train the main model. Twin: training set is split into two halves. Each half is used to "
        "train a separate il model. Each il model provides the irreducible loss for the samples that the"
        "model is not trained on. The original training set is used to train the main model."
    )
    holdout_set_ratio: int = Field(
        description="How much of the training set is used as the holdout set.",
        min=0,
        max=100,
    )
    holdout_set_ratio_max: int = Field(
        description="Reference maximum holdout_set_ratio value. Defaults to 100, which implies percent."
        " If you set this to 1000, holdout_set_ratio describes promille instead.",
        default=100,
        min=1,
    )
    il_training_config: ILTrainingConfig = Field(description="The configuration for the IL training.")

    @model_validator(mode="after")
    def validate_holdout_set_ratio(self) -> Self:
        if self.holdout_set_strategy == "Twin" and self.holdout_set_ratio != 50:
            raise ValueError("holdout_set_ratio should be 100 for the Twin strategy.")
        if self.holdout_set_ratio > self.holdout_set_ratio_max:
            raise ValueError("holdout_set_ratio cannot be greater than holdout_set_ratio_max.")
        return self


class RS2DownsamplingConfig(BaseDownsamplingConfig):
    """Config for the RS2 downsampling strategy."""

    strategy: Literal["RS2"] = "RS2"
    period: Literal[1] = 1  # RS2 needs to sample every epoch
    sample_then_batch: Literal[True] = True  # RS2 only supports StB
    with_replacement: bool = Field(
        description=(
            "Whether we resample from the full TTS each epoch (= True) or train "
            "on all the data with a different subset each epoch (= False)."
        )
    )


SingleDownsamplingConfig = Annotated[
    UncertaintyDownsamplingConfig
    | PerTokenUncertaintyDownsamplingConfig
    | KcenterGreedyDownsamplingConfig
    | GradMatchDownsamplingConfig
    | CraigDownsamplingConfig
    | LossDownsamplingConfig
    | SubmodularDownsamplingConfig
    | GradNormDownsamplingConfig
    | NoDownsamplingConfig
    | RHOLossDownsamplingConfig
    | RS2DownsamplingConfig,
    Field(discriminator="strategy"),
]


class MultiDownsamplingConfig(ModynBaseModel):
    downsampling_list: list[SingleDownsamplingConfig] = Field(description="An array of downsampling strategies.")
    downsampling_thresholds: list[int] = Field(
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
