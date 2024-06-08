from __future__ import annotations

from functools import cached_property
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

from modyn.config.schema.modyn_base_model import ModynBaseModel
from modyn.config.schema.sampling.downsampling_config import (
    MultiDownsamplingConfig,
    NoDownsamplingConfig,
    SingleDownsamplingConfig,
)
from modyn.config.schema.training.training_config import TrainingConfig
from modyn.supervisor.internal.eval_strategies import OffsetEvalStrategy
from modyn.utils import validate_timestr
from modyn.utils.utils import SECONDS_PER_UNIT, deserialize_function
from pydantic import Field, NonNegativeInt, field_validator, model_validator
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


# ------------------------------------------------------- Data ------------------------------------------------------- #


class DataConfig(ModynBaseModel):
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
    label_transformer_function: str = Field(
        "", description="Function used to transform the label (tensors of integers)."
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


class TimeTriggerConfig(ModynBaseModel):
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


# ---------------------------------------------------- EVALUATION ---------------------------------------------------- #


class Metric(ModynBaseModel):
    name: str = Field(description="The name of the evaluation metric.")
    config: dict[str, Any] = Field({}, description="Configuration for the evaluation metric.")
    evaluation_transformer_function: str | None = Field(
        "",
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

    @model_validator(mode="after")
    def can_instantiate_metric(self) -> Self:
        # We have to import the MetricFactory here to avoid issues with the multiprocessing context
        # If we move it up, then we'll have `spawn` everywhere, and then the unit tests on Github
        # are way too slow.
        # pylint: disable-next=wrong-import-position,import-outside-toplevel
        from modyn.evaluator.internal.metric_factory import MetricFactory  # fmt: skip  # noqa  # isort:skip
        try:
            MetricFactory.get_evaluation_metric(self.name, self.evaluation_transformer_function, self.config)
        except NotImplementedError as exc:
            raise ValueError(f"Cannot instantiate metric {self.name}!") from exc

        return self


class MatrixEvalStrategyConfig(ModynBaseModel):
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


class OffsetEvalStrategyConfig(ModynBaseModel):
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


class MatrixEvalStrategyModel(ModynBaseModel):
    name: Literal["MatrixEvalStrategy"]
    config: MatrixEvalStrategyConfig


class OffsetEvalStrategyModel(ModynBaseModel):
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


class ResultWriter(ModynBaseModel):
    name: str = Field(description="The name of the result writer.")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration for the result writer.")


ResultWriterType = Literal["json", "json_dedicated", "tensorboard"]
"""
- json: appends the evaluations to the standard json logfile.
- json_dedicated: dumps the results into dedicated json files for each evaluation.
- tensorboard: output the evaluation to dedicated tensorboard files."""


class EvaluationConfig(ModynBaseModel):
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
