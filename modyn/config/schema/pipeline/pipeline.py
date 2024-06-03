from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from modyn.config.schema.modyn_base_model import ModynBaseModel
from modyn.config.schema.pipeline.base import REGEX_TIME_UNIT
from modyn.config.schema.pipeline.data import DataConfig
from modyn.config.schema.pipeline.evaluation import EvaluationConfig
from modyn.config.schema.sampling.downsampling_config import (
    MultiDownsamplingConfig,
    NoDownsamplingConfig,
    SingleDownsamplingConfig,
)
from modyn.utils.utils import SECONDS_PER_UNIT
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

# ----------------------------------------------------- PIPELINE ----------------------------------------------------- #


class Pipeline(ModynBaseModel):
    name: str = Field(description="The name of the pipeline.")
    description: Optional[str] = Field(None, description="The description of the pipeline.")
    version: Optional[str] = Field(None, description="The version of the pipeline.")


# ------------------------------------------------------- MODEL ------------------------------------------------------ #


class ModelConfig(ModynBaseModel):
    id: str = Field(description="The ID of the model that should be trained.")
    config: dict = Field(
        default_factory=dict,
        description="Configuration dictionary that will be passed to the model on initialization.",
    )


# --------------------------------------------------- MODEL STORAGE -------------------------------------------------- #


class _BaseModelStrategy(ModynBaseModel):
    """Base class for the model storage strategies."""

    config: Dict[str, Any] = Field(
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


# ----------------------------------------------------- TRAINING ----------------------------------------------------- #

LimitResetStrategy = Literal["lastX", "sampleUAR"]


class PresamplingConfig(ModynBaseModel):
    """Config for the presampling strategy of CoresetStrategy. If missing, no presampling is applied."""

    strategy: Literal["Random", "RandomNoReplacement", "LabelBalanced", "TriggerBalanced", "No"] = Field(
        description="Strategy used to presample the data."
        "Only the prefix, i.e. without `PresamplingStrategy`, is needed."
    )
    ratio: int = Field(
        description="Percentage of points on which the metric (loss, gradient norm,..) is computed.",
        min=0,
        max=100,
    )
    force_column_balancing: bool = Field(False)
    force_required_target_size: bool = Field(False)


StorageBackend = Literal["database", "local"]


class _BaseSelectionStrategy(ModynBaseModel):
    maximum_keys_in_memory: int = Field(
        description="Limits how many keys should be materialized at a time in the strategy.",
        ge=1,
    )
    processor_type: Optional[str] = Field(
        None,
        description="The name of the Metadata Processor strategy that should be used.",
    )
    limit: int = Field(
        -1,
        description="This limits how many data points we train on at maximum on a trigger. Set to -1 to disable limit.",
    )
    storage_backend: StorageBackend = Field(
        "database",
        description="Most strategies currently support `database`, and the NewDataStrategy supports `local` as well.",
    )
    uses_weights: bool = Field(
        False,
        description=(
            "If set to true, weights are supplied from the selector and weighted SGD is used to train. Please note "
            "that this option is not related to Downsampling strategy, where weights are computed during training."
        ),
    )
    tail_triggers: int | None = Field(
        description=(
            "For the training iteration, just use data from this trigger and the previous tail_triggers. "
            "If tail_triggers = 0, it means we reset after every trigger. "
            "If tail_triggers = None, it means we use all data."
        ),
    )

    @model_validator(mode="after")
    def validate_tail_triggers(self) -> Self:
        if self.tail_triggers is not None and self.tail_triggers < 0:
            raise ValueError("tail_triggers must be a non-negative integer or None.")
        return self


class FreshnessSamplingStrategyConfig(_BaseSelectionStrategy):
    name: Literal["FreshnessSamplingStrategy"] = Field("FreshnessSamplingStrategy")
    unused_data_ratio: int = Field(
        description=(
            "Ratio that defines how much data in the training set per trigger should be from previously unused "
            "data (in all previous triggers)."
        ),
        ge=1,
        le=99,
    )


class NewDataStrategyConfig(_BaseSelectionStrategy):
    name: Literal["NewDataStrategy"] = Field("NewDataStrategy")

    limit_reset: LimitResetStrategy = Field(
        "lastX",
        description=(
            "Strategy to follow for respecting the limit in case of reset. Only used when tail_triggers == 0."
        ),
    )


class CoresetStrategyConfig(_BaseSelectionStrategy):
    name: Literal["CoresetStrategy"] = Field("CoresetStrategy")

    presampling_config: PresamplingConfig = Field(
        PresamplingConfig(strategy="No", ratio=100),
        description=("Config for the presampling strategy. If missing, no presampling is applied."),
    )
    downsampling_config: SingleDownsamplingConfig | MultiDownsamplingConfig = Field(
        NoDownsamplingConfig(strategy="No", ratio=100),
        description="Configurates the downsampling with one or multiple strategies.",
    )


SelectionStrategy = Annotated[
    Union[FreshnessSamplingStrategyConfig, NewDataStrategyConfig, CoresetStrategyConfig], Field(discriminator="name")
]


class CheckpointingConfig(ModynBaseModel):
    """Configuration for checkpointing during training."""

    activated: bool = Field(description="Whether we checkpoint or not.")
    interval: int | None = Field(None, description="In what interval we checkpoint.")
    path: Path | None = Field(
        None,
        description="The path on the training node where the checkpoints are stored.",
    )


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


class TrainingConfig(ModynBaseModel):
    gpus: int = Field(description="The number of GPUs that should be used for training.", ge=1)
    epochs_per_trigger: int = Field(1, description="The number of epochs that should be trained per trigger.")
    num_samples_to_pass: list[int] | None = Field(
        None,
        description=(
            "If it is set, during the i-th trigger, the number of samples to train on (multiple passes counted as "
            "exactly multiple times, e.g. if we train for 2 epochs then each sample is counted twice) is the i-th "
            "element in the array. If the array is not long enough, out-of-range triggers will fall back to "
            "the `epochs_per_trigger` constraint. `epochs_per_trigger` has to effect to in-range triggers."
        ),
        min_length=1,
    )
    num_prefetched_partitions: int = Field(
        1,
        description="The number of partitions that are prefetched per DataLoader worker.",
    )
    parallel_prefetch_requests: int = Field(
        1, description="The number of parallel prefetch requests per DataLoader worker."
    )
    device: str = Field(
        "cpu",
        description="The device the model should be put on.",
        pattern=r"^(cpu|cuda:\d+)$",
    )
    amp: bool = Field(False, description="Whether to use automatic mixed precision.")
    dataloader_workers: int = Field(
        description="The number of data loader workers on the trainer node that fetch data from storage.", ge=1
    )
    batch_size: int = Field(description="The batch size to be used during training.", ge=1)
    use_previous_model: bool = Field(
        description=(
            "If True, on trigger, we continue training on the model outputted by the previous trigger. If False, "
            "we start with random weights. If initial_model is 'pretrained', cannot be False."
        )
    )
    seed: Optional[int] = Field(
        None,
        description=(
            "If provided, every random python function (torch, numpy..) is seeded with this number. Must be in the "
            "range [0,100]. Please be aware that if you want to seed postgres you must specify its seed in the "
            "modyn_config."
        ),
    )
    initial_model: Literal["random", "pretrained"] = Field(
        description="What type of initial model should be used (random or pretrained)."
    )
    initial_model_id: Optional[int] = Field(
        None,
        description="The ID of the model that should be used as the initial model.",
    )
    checkpointing: CheckpointingConfig = Field(description="Configuration of checkpointing during training")
    optimizers: List[OptimizerConfig] = Field(
        description="An array of the optimizers for the training",
        min_length=1,
    )
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

    # [Additional validation]

    @field_validator("gpus")
    @classmethod
    def validate_gpus(cls, value: int) -> int:
        if value > 1:
            raise ValueError("Currently, only single GPU training is supported.")
        return value

    @model_validator(mode="after")
    def validate_pretrained(self) -> Self:
        if self.initial_model == "pretrained":
            if not self.use_previous_model:
                raise ValueError(
                    "Cannot have use_previous_model == False and use a pretrained initial model."
                    "Initial model would get lost after first trigger."
                )
            if not self.initial_model_id:
                raise ValueError("Initial model set to pretrained, but no initial_model_id given")
        return self


class TimeTriggerConfig(ModynBaseModel):
    id: Literal["TimeTrigger"] = Field("TimeTrigger")
    every: str = Field(
        description="Interval length for the trigger as an integer followed by a time unit: s, m, h, d, w, y",
        pattern=rf"^\d+{REGEX_TIME_UNIT}$",
    )
    sample_size: int | None = Field(None, description="The number of samples used for the metric calculation.", ge=1)
    metric: str = Field("model", description="The metric used for drift detection.")
    metric_config: dict[str, Any] = Field(default_factory=dict, description="Configuration for the evidently metric.")

    @cached_property
    def every_seconds(self) -> int:
        unit = str(self.every)[-1:]
        num = int(str(self.every)[:-1])
        return num * SECONDS_PER_UNIT[unit]


class DataAmountTriggerConfig(ModynBaseModel):
    id: Literal["DataAmountTrigger"] = Field("DataAmountTrigger")
    num_samples: int = Field(description="The number of samples that should trigger the pipeline.", ge=1)


class DataDriftTriggerConfig(ModynBaseModel):
    id: Literal["DataDriftTrigger"] = Field("DataDriftTrigger")
    detection_interval_data_points: int = Field(
        1000, description="The number of samples in the interval after which drift detection is performed.", ge=1
    )
    sample_size: int | None = Field(None, description="The number of samples used for the metric calculation.", ge=1)
    metric: str = Field("model", description="The metric used for drift detection.")
    metric_config: dict[str, Any] = Field(default_factory=dict, description="Configuration for the evidently metric.")


TriggerConfig = Annotated[
    Union[TimeTriggerConfig, DataAmountTriggerConfig, DataDriftTriggerConfig], Field(discriminator="id")
]

# ----------------------------------------------------- PIPELINE ----------------------------------------------------- #


class ModynPipelineConfig(ModynBaseModel):
    pipeline: Pipeline

    # model is a reserved keyword in Pydantic, so we use modyn_model instead
    modyn_model: ModelConfig = Field(alias="model")
    modyn_model_storage: PipelineModelStorageConfig = Field(alias="model_storage")

    training: TrainingConfig
    data: DataConfig
    trigger: TriggerConfig
    selection_strategy: SelectionStrategy
    evaluation: EvaluationConfig | None = Field(None)
