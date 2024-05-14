from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# ----------------------------------------------------- PIPELINE ----------------------------------------------------- #


class Pipeline(BaseModel):
    name: str = Field(description="The name of the pipeline.")
    description: Optional[str] = Field(None, description="The description of the pipeline.")
    version: Optional[str] = Field(None, description="The version of the pipeline.")


# ------------------------------------------------------- MODEL ------------------------------------------------------ #


class ModelConfig(BaseModel):
    id: str = Field(description="The ID of the model that should be trained.")
    config: dict = Field(
        default_factory=dict,
        description="Configuration dictionary that will be passed to the model on initialization.",
    )


# --------------------------------------------------- MODEL STORAGE -------------------------------------------------- #


class FullModelStrategy(BaseModel):
    """Which full model strategy is used for the model storage."""

    name: str = Field(
        description="Name of the full model strategy. We currently support PyTorchFullModel and BinaryFullModel."
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration dictionary that will be passed to the strategy.",
    )
    zip: bool = Field(False, description="Whether to zip the file in the end.")
    zip_algorithm: str = Field(default="ZIP_DEFLATED", description="Which zip algorithm to use.")


class IncrementalModelStrategy(BaseModel):
    """Which incremental model strategy is used for the model storage."""

    name: str = Field(description="Name of the incremental model strategy. We currently support WeightsDifference.")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration dictionary that will be passed to the strategy.",
    )
    zip: Optional[bool] = Field(False, description="Whether to zip the file in the end.")
    zip_algorithm: str = Field(
        default="ZIP_DEFLATED",
        description="Which zip algorithm to use. Default is ZIP_DEFLATED.",
    )
    full_model_interval: Optional[float] = Field(
        None, description="In which interval we are using the full model strategy."
    )


class PipelineModelStorageConfig(BaseModel):
    full_model_strategy: FullModelStrategy = Field(description="Which full model strategy is used.")
    incremental_model_strategy: Optional[IncrementalModelStrategy] = Field(
        None, description="Which incremental model strategy is used."
    )


# ----------------------------------------------------- TRAINING ----------------------------------------------------- #

LimitResetStrategy = Literal["lastX", "sampleUAR"]


class PresamplingConfig(BaseModel):
    """Config for the presampling strategy of CoresetStrategy. If missing, no presampling is applied."""

    strategy: Literal["Random", "RandomNoReplacement", "LabelBalanced", "TriggerBalanced", "No"] = Field(
        description="Strategy used to presample the data."
    )
    ratio: float = Field(
        description="Percentage of points on which the metric (loss, gradient norm,..) is computed.",
        min=0,
        max=100,
    )


class DownsamplingConfig(BaseModel):
    """Config for the downsampling strategy of SelectionStrategy."""

    strategy: Literal["Random", "RandomNoReplacement", "LabelBalanced", "TriggerBalanced", "No"] = Field(
        description="Strategy used to downsample the data. Available strategies: Loss, Gradnorm, No (no downsampling)."
    )
    sample_then_batch: bool = Field(
        False,
        description=(
            "If True, the samples are first sampled and then batched and supplied to the training loop. If False, "
            "the datapoints are first divided into batches and then sampled."
        ),
    )
    ratio: float = Field(
        description="Ratio post_sampling_size/pre_sampling_size. E.g. with 160 records and a ratio of 50 we keep 80.",
        min=0,
        max=100,
    )
    period: int = Field(
        1,
        description=(
            "In multi-epoch training and sample_then_batch, how frequently the data is selected. "
            "`1` selects every epoch. To select once per trigger, set this parameter to 0."
        ),
        min=0,
    )


class MultiDownsamplingConfig(BaseModel):
    downsampling_list: List[DownsamplingConfig] = Field(description="An array of downsampling strategies.")
    downsampling_thresholds: List[int] = Field(
        description=(
            "A list of thresholds to switch from a downsampler to another. The i-th threshold is used for the "
            "transition from the i-th downsampler to the (i+1)-th. This array should have one less item on the list "
            "of downsamplers."
        )
    )

    @model_validator(mode="after")
    def validate_downsampling_thresholds(
        self: "MultiDownsamplingConfig",
    ) -> "MultiDownsamplingConfig":
        if len(self.downsampling_thresholds) != len(self.downsampling_list) - 1:
            raise ValueError("The downsampling_thresholds list should have one less item than the downsampling_list.")
        return self


StorageBackend = Literal["database", "local"]


class _BaseSelectionStrategyConfig(BaseModel):
    limit: int = Field(
        description="This limits how many data points we train on at maximum on a trigger. Set to -1 to disable limit."
    )
    storage_backend: StorageBackend = Field(
        description="Most strategies currently support `database`, and the NewDataStrategy supports `local` as well."
    )
    uses_weights: bool = Field(
        False,
        description=(
            "If set to true, weights are supplied from the selector and weighted SGD is used to train. Please note "
            "that this option is not related to Downsampling strategy, where weights are computed during training."
        ),
    )
    reset_after_trigger: bool = Field(
        False,
        description="If set to true, the selection strategy resets its internal state after a trigger.",
    )
    tail_triggers: int = Field(
        0,
        description=(
            "For the training iteration, just use data from this trigger and the previous tail_triggers. "
            "reset_after_trigger is equivalent to tail_triggers = 0. Omit this parameter if you want to use every "
            "previous datapoint."
        ),
    )


class FreshnessSamplingStrategyConfig(_BaseSelectionStrategyConfig):
    unused_data_ratio: float = Field(
        0.0,
        description=(
            "Ratio that defines how much data in the training set per trigger should be from previously unused "
            "data (in all previous triggers)."
        ),
    )


class NewDataSelectionStrategyConfig(_BaseSelectionStrategyConfig):
    limit_reset: LimitResetStrategy = Field(
        description=(
            "Strategy to follow for respecting the limit in case of reset. Only used when reset_after_trigger == true."
        )
    )


class CoresetSelectionStrategyConfig(_BaseSelectionStrategyConfig):
    presampling_config: Optional[PresamplingConfig] = Field(
        None,
        description=("Config for the presampling strategy. If missing, no presampling is applied."),
    )
    downsampling_config: DownsamplingConfig | MultiDownsamplingConfig = Field(
        description="Configurates the downsampling with one or multiple strategies."
    )


SelectionStrategyConfig = Union[
    FreshnessSamplingStrategyConfig, NewDataSelectionStrategyConfig, CoresetSelectionStrategyConfig
]


class SelectionStrategy(BaseModel):
    name: str = Field(description="The name of the selection strategy that should be used.")
    maximum_keys_in_memory: int = Field(
        description="Limits how many keys should be materialized at a time in the strategy."
    )
    config: Optional[SelectionStrategyConfig] = Field(None, description="Configuration for the selection strategy.")
    processor_type: Optional[str] = Field(
        None,
        description="The name of the Metadata Processor strategy that should be used.",
    )


class CheckpointingConfig(BaseModel):
    """Configuration for checkpointing during training."""

    activated: bool = Field(description="Whether we checkpoint or not.")
    interval: int | None = Field(None, description="In what interval we checkpoint.")
    path: Path | None = Field(
        None,
        description="The path on the training node where the checkpoints are stored.",
    )


OptimizerSource = Literal["PyTorch", "APEX"]


class OptimizerParamGroup(BaseModel):
    """Configuration for a parameter group."""

    module: str = Field(description="A set of parameters.")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional configuration for the parameter group. (e.g. learning rate)",
    )


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer used during training."""

    name: str = Field(description="The name of the optimizer (like an ID).")
    algorithm: str = Field(description="The type of the optimizer (e.g. SGD).")
    source: OptimizerSource = Field(description="The framework/package the optimizer comes from.")
    param_groups: List[OptimizerParamGroup] = Field(
        description="An array of the parameter groups (parameters and optional configs) this optimizer supervises.",
        min_length=1,
    )


class OptimizationCriterion(BaseModel):
    """Configuration for the optimization criterion that we optimize."""

    name: str = Field(description="The name of the criterion that the pipeline uses (e.g., CrossEntropyLoss).")
    config: Dict[str, Any] = Field(default_factory=dict, description="Optional configuration of the criterion.")


LrSchedulerSource = Literal["PyTorch", "custom"]


class LrSchedulerConfig(BaseModel):
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
    def validate_optimizers(self: "LrSchedulerConfig") -> "LrSchedulerConfig":
        if self.source == "PyTorch" and len(self.optimizers) != 1:
            raise ValueError("In case a PyTorch LR scheduler is used, the optimizers list should have only one item.")
        return self


class TrainingConfig(BaseModel):
    gpus: int = Field(description="The number of GPUs that should be used for training.")
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
    dataloader_workers: float = Field(
        description="The number of data loader workers on the trainer node that fetch data from storage."
    )
    batch_size: float = Field(description="The batch size to be used during training.")
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
    selection_strategy: SelectionStrategy = Field(description="Configuration for the Selector")

    initial_model: str = Field(description="What type of initial model should be used (random or pretrained).")
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


# ---------------------------------------------------- EVALUATION ---------------------------------------------------- #


class DataConfig(BaseModel):
    dataset_id: str = Field(description="ID of dataset to be used for training.")
    bytes_parser_function: str = Field(
        description=(
            "Function used to convert bytes received from the Storage, to a format useful for further transformations "
            "(e.g. Tensors) This function is called before any other transformations are performed on the data."
        )
    )
    transformations: List[Any] = Field(
        default_factory=list,
        description=(
            "Further transformations to be applied on the data after bytes_parser_function has been applied."
            "For example, this can be torchvision transformations."
        ),
    )
    label_transformer_function: Optional[str] = Field(
        None, description="Function used to transform the label (tensors of integers)."
    )
    tokenizer: Optional[str] = Field(
        None,
        description="Function to tokenize the input. Must be a class in modyn.models.tokenizers.",
    )


# ------------------------------------------------------ TRIGGER ----------------------------------------------------- #


class TriggerConfig(BaseModel):
    id: str = Field(description="Type of trigger to be used.")

    trigger_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration dictionary that will be passed to the trigger on initialization.",
    )


# ---------------------------------------------------- EVALUATION ---------------------------------------------------- #


class Metric(BaseModel):
    name: str = Field(description="The name of the evaluation metric.")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration for the evaluation metric.")
    evaluation_transformer_function: Optional[str] = Field(
        None,
        description="A function used to transform the model output before evaluation.",
    )


class DatasetConfig(BaseModel):
    dataset_id: str = Field(description="The id of the dataset.")
    bytes_parser_function: str = Field(
        description=(
            "Function used to convert bytes received from the storage, to a format useful for further transformations "
            "(e.g. Tensors). This function is called before any other transformations are performed on the data."
        )
    )
    transformations: List[Any] = Field(
        default_factory=list,
        description=(
            "Further (optional) transformations to be applied on the data after bytes_parser_function has been "
            "applied. For example, this can be torchvision transformations."
        ),
    )
    label_transformer_function: Optional[str] = Field(
        None,
        description="function used to transform the label which are tensors of integers",
    )
    batch_size: int = Field(description="The batch size to be used during evaluation.")
    dataloader_workers: float = Field(
        description="The number of data loader workers on the evaluation node that fetch data from storage."
    )
    metrics: List[Metric] = Field(
        description="All metrics used to evaluate the model on the given dataset.",
        min_length=1,
    )


class ResultWriter(BaseModel):
    name: str = Field(description="The name of the result writer.")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration for the result writer.")


class EvaluationConfig(BaseModel):
    device: str = Field(description="The device the model should be put on.")
    result_writers: List[str] = Field(
        description=(
            "List of names that specify in which formats to store the evaluation results. We currently support "
            "json and tensorboard."
        ),
        min_length=1,
    )
    datasets: List[DatasetConfig] = Field(
        description="An array of all datasets on which the model is evaluated.",
        min_length=1,
    )


# ----------------------------------------------------- PIPELINE ----------------------------------------------------- #


class ModynPipelineConfig(BaseModel):
    pipeline: Pipeline

    # model is a reserved keyword in Pydantic, so we use modyn_model instead
    modyn_model: ModelConfig = Field(alias="model")
    modyn_model_storage: PipelineModelStorageConfig = Field(alias="model_storage")

    training: TrainingConfig
    data: DataConfig
    trigger: TriggerConfig
    evaluation: EvaluationConfig | None = Field(None)
