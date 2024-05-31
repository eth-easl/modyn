from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

from modyn.supervisor.internal.eval_strategies import OffsetEvalStrategy
from modyn.utils import validate_timestr
from modyn.utils.utils import SECONDS_PER_UNIT, deserialize_function
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, NonNegativeInt, field_validator, model_validator
from typing_extensions import Self

# ----------------------------------------------------- PIPELINE ----------------------------------------------------- #


class BaseModel(PydanticBaseModel):
    class Config:
        extra = "forbid"


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


class _BaseModelStrategy(BaseModel):
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
        "Only the prefix, i.e. without `PresamplingStrategy`, is needed."
    )
    ratio: int = Field(
        description="Percentage of points on which the metric (loss, gradient norm,..) is computed.",
        min=0,
        max=100,
    )
    force_column_balancing: bool = Field(False)
    force_required_target_size: bool = Field(False)


class DownsamplingConfig(BaseModel):
    """Config for the downsampling strategy of SelectionStrategy."""

    class Config:
        # depending on the strategy, additional fields may be required. Please refer to the respective strategy class
        # for exact configurable fields.
        extra = "allow"

    strategy: Literal[
        "Craig", "GradMatch", "GradNorm", "KcenterGreedy", "Loss", "No", "Submodular", "Uncertainty"
    ] = Field(
        description="Strategy used to downsample the data."
        "Only the prefix, i.e. without `DownsamplingStrategy`, is needed."
    )
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
    period: int | None = Field(
        None,
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


StorageBackend = Literal["database", "local"]


class _BaseSelectionStrategy(BaseModel):
    maximum_keys_in_memory: int = Field(
        description="Limits how many keys should be materialized at a time in the strategy."
    )
    processor_type: Optional[str] = Field(
        None,
        description="The name of the Metadata Processor strategy that should be used.",
    )
    limit: int = Field(
        -1,
        description="This limits how many data points we train on at maximum on a trigger. Set to -1 to disable limit.",
    )
    reset_after_trigger: bool = Field(
        False,
        description="If set to true, the selection strategy resets its internal state after a trigger.",
    )
    storage_backend: StorageBackend | None = Field(
        None,
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
        None,
        description=(
            "For the training iteration, just use data from this trigger and the previous tail_triggers. "
            "reset_after_trigger is equivalent to tail_triggers = 0. Omit this parameter if you want to use every "
            "previous datapoint."
        ),
    )


class FreshnessSamplingStrategy(_BaseSelectionStrategy):
    name: Literal["FreshnessSamplingStrategy"] = Field("FreshnessSamplingStrategy")

    unused_data_ratio: float = Field(
        0.0,
        description=(
            "Ratio that defines how much data in the training set per trigger should be from previously unused "
            "data (in all previous triggers)."
        ),
    )


class NewDataSelectionStrategy(_BaseSelectionStrategy):
    name: Literal["NewDataStrategy"] = Field("NewDataStrategy")

    limit_reset: LimitResetStrategy = Field(
        "lastX",
        description=(
            "Strategy to follow for respecting the limit in case of reset. Only used when reset_after_trigger == true."
        ),
    )


class CoresetSelectionStrategy(_BaseSelectionStrategy):
    name: Literal["CoresetStrategy"] = Field("CoresetStrategy")

    presampling_config: Optional[PresamplingConfig] = Field(
        None,
        description=("Config for the presampling strategy. If missing, no presampling is applied."),
    )
    downsampling_config: DownsamplingConfig | MultiDownsamplingConfig | None = Field(
        None, description="Configurates the downsampling with one or multiple strategies."
    )


SelectionStrategy = Union[FreshnessSamplingStrategy, NewDataSelectionStrategy, CoresetSelectionStrategy]


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


LrSchedulerSource = Literal["PyTorch", "Custom"]


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
    def validate_optimizers(self) -> Self:
        if self.source == "PyTorch" and len(self.optimizers) != 1:
            raise ValueError("In case a PyTorch LR scheduler is used, the optimizers list should have only one item.")
        return self


class TrainingConfig(BaseModel):
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
    shuffle: bool = Field(
        description=(
            "If True, we shuffle the order of partitions and the data within each partition at each worker."
            "Otherwise, the output order is deterministic."
        )
    )
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


# ------------------------------------------------------- Data ------------------------------------------------------- #


class DataConfig(BaseModel):
    dataset_id: str = Field(description="ID of dataset to be used.")
    bytes_parser_function: str = Field(
        description=(
            "Function used to convert bytes received from the Storage, to a format useful for further transformations "
            "(e.g. Tensors) This function is called before any other transformations are performed on the data."
        )
    )
    transformations: List[str] = Field(
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

    @field_validator("bytes_parser_function", mode="before")
    @classmethod
    def validate_bytes_parser_function(cls, value: str) -> str:
        try:
            res = deserialize_function(value, "bytes_parser_function")
            if not callable(res):
                raise ValueError("Function 'bytes_parser_function' must be callable!")
        except AttributeError as exc:
            raise ValueError("Function 'bytes_parser_function' could not be parsed!") from exc
        return value

    @property
    def bytes_parser_function_deserialized(self) -> Callable:
        func = deserialize_function(self.bytes_parser_function, "bytes_parser_function")
        if func is None:
            raise ValueError("Function 'bytes_parser_function' could not be parsed!")
        return func


# ------------------------------------------------------ TRIGGER ----------------------------------------------------- #

_REGEX_TIME_UNIT = r"(s|m|h|d|w|y)"


class TimeTriggerConfig(BaseModel):
    id: Literal["TimeTrigger"] = Field("TimeTrigger")
    every: str = Field(
        description="Interval length for the trigger as an integer followed by a time unit: s, m, h, d, w, y",
        pattern=rf"^\d+{_REGEX_TIME_UNIT}$",
    )

    @cached_property
    def every_seconds(self) -> int:
        unit = str(self.every)[-1:]
        num = int(str(self.every)[:-1])
        return num * SECONDS_PER_UNIT[unit]


class DataAmountTriggerConfig(BaseModel):
    id: Literal["DataAmountTrigger"] = Field("DataAmountTrigger")
    num_samples: int = Field(description="The number of samples that should trigger the pipeline.", ge=1)


class DataDriftTriggerConfig(BaseModel):
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


# ---------------------------------------------------- EVALUATION ---------------------------------------------------- #


class Metric(BaseModel):
    name: str = Field(description="The name of the evaluation metric.")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration for the evaluation metric.")
    evaluation_transformer_function: str | None = Field(
        None,
        description="A function used to transform the model output before evaluation.",
    )

    @field_validator("evaluation_transformer_function", mode="before")
    @classmethod
    def validate_evaluation_transformer_function(cls, value: str) -> str | None:
        if not value:
            return None
        try:
            deserialize_function(value, "evaluation_transformer_function")
        except AttributeError as exc:
            raise ValueError("Function 'evaluation_transformer_function' could not be parsed!") from exc
        return value

    @property
    def evaluation_transformer_function_deserialized(self) -> Callable | None:
        if self.evaluation_transformer_function:
            return deserialize_function(self.evaluation_transformer_function, "evaluation_transformer_function")
        return None


class MatrixEvalStrategyConfig(BaseModel):
    eval_every: str = Field(
        description="The interval length for the evaluation "
        "specified by an integer followed by a time unit (e.g. '100s')."
    )
    eval_start_from: NonNegativeInt = Field(
        description="The timestamp from which the evaluation should start (inclusive). This timestamp is in seconds."
    )
    eval_end_at: NonNegativeInt = Field(
        description="The timestamp at which the evaluation should end (exclusive). This timestamp is in seconds."
    )

    @field_validator("eval_every")
    @classmethod
    def validate_eval_every(cls, value: str) -> str:
        if not validate_timestr(value):
            raise ValueError("eval_every must be a valid time string")
        return value

    @model_validator(mode="after")
    def eval_end_at_must_be_larger(self) -> Self:
        if self.eval_start_from >= self.eval_end_at:
            raise ValueError("eval_end_at must be larger than eval_start_from")
        return self


class OffsetEvalStrategyConfig(BaseModel):
    offsets: List[str] = Field(
        description=(
            "A list of offsets that define the evaluation intervals. For valid offsets, see the class docstring of "
            "OffsetEvalStrategy."
        ),
        min_length=1,
    )

    @field_validator("offsets")
    @classmethod
    def validate_offsets(cls, value: List[str]) -> List[str]:
        for offset in value:
            if offset not in [OffsetEvalStrategy.NEGATIVE_INFINITY, OffsetEvalStrategy.INFINITY]:
                if not validate_timestr(offset):
                    raise ValueError(f"offset {offset} must be a valid time string")
        return value


class MatrixEvalStrategyModel(BaseModel):
    name: Literal["MatrixEvalStrategy"]
    config: MatrixEvalStrategyConfig


class OffsetEvalStrategyModel(BaseModel):
    name: Literal["OffsetEvalStrategy"]
    config: OffsetEvalStrategyConfig


EvalStrategyModel = Annotated[Union[MatrixEvalStrategyModel, OffsetEvalStrategyModel], Field(discriminator="name")]


class EvalDataConfig(DataConfig):
    batch_size: int = Field(description="The batch size to be used during evaluation.", ge=1)
    dataloader_workers: int = Field(
        description="The number of data loader workers on the evaluation node that fetch data from storage.", ge=1
    )
    metrics: List[Metric] = Field(
        description="All metrics used to evaluate the model on the given dataset.",
        min_length=1,
    )
    tokenizer: Optional[str] = Field(
        None,
        description="Function to tokenize the input. Must be a class in modyn.models.tokenizers.",
    )


class ResultWriter(BaseModel):
    name: str = Field(description="The name of the result writer.")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration for the result writer.")


ResultWriterType = Literal["json", "json_dedicated", "tensorboard"]
"""
- json: appends the evaluations to the standard json logfile.
- json_dedicated: dumps the results into dedicated json files for each evaluation.
- tensorboard: output the evaluation to dedicated tensorboard files."""


class EvaluationConfig(BaseModel):
    eval_strategy: EvalStrategyModel = Field(description="The evaluation strategy that should be used.")
    device: str = Field(description="The device the model should be put on.")
    result_writers: List[ResultWriterType] = Field(
        ["json"],
        description=(
            "List of names that specify in which formats to store the evaluation results. We currently support "
            "json and tensorboard."
        ),
        min_length=1,
    )
    datasets: List[EvalDataConfig] = Field(
        description="An array of all datasets on which the model is evaluated.",
        min_length=1,
    )

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: List[EvalDataConfig]) -> List[EvalDataConfig]:
        dataset_ids = [dataset.dataset_id for dataset in value]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Dataset IDs must be unique.")
        return value


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
