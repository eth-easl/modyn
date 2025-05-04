from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from modyn.config.schema.base_model import ModynBaseModel

from .downsampling_config import MultiDownsamplingConfig, NoDownsamplingConfig, SingleDownsamplingConfig

LimitResetStrategy = Literal["lastX", "sampleUAR"]


class PresamplingConfig(ModynBaseModel):
    """Config for the presampling strategy of CoresetStrategy.

    If missing, no presampling is applied.
    """

    strategy: Literal["Random", "RandomNoReplacement", "LabelBalanced", "TriggerBalanced", "No"] = Field(
        description="Strategy used to presample the data."
        "Only the prefix, i.e. without `PresamplingStrategy`, is needed."
    )
    ratio: int = Field(
        description=(
            "Ratio of points on which the metric (loss, gradient norm,..) is computed."
            "By default with ratio_max=100, this describes the selection ratio in percent."
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
    custom_prompt: str | None = Field(
        None, description="Optional custom prompt to override the default evaluation prompt for the LLM."
    )
    api_key: str = Field("", description="API key for the LLM service. Required if strategy is LLMEvaluation.")
    base_url: str = Field("https://fmapi.swissai.cscs.ch", description="Base URL for the LLM service.")
    model_name: str = Field("meta-llama/Llama-3.3-70B-Instruct", description="Model name for the LLM service.")
    dataset_id: str | None = Field(
        None, description="Dataset ID for the LLM service. Must be provided if we use LLMEvaluation."
    )
    batch_size: int = Field(10, description="Batch size to use during LLM evaluation.")
    force_column_balancing: bool = Field(False)
    force_required_target_size: bool = Field(False)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.ratio > self.ratio_max:
            raise ValueError("ratio cannot be greater than ratio_max.")

        if self.strategy == "LLMEvaluation":
            if not self.dataset_id:
                raise ValueError("dataset_id must be provided when strategy is 'LLMEvaluation'.")
            if not self.api_key:
                raise ValueError("api_key must be provided when strategy is 'LLMEvaluation'.")

        return self


StorageBackend = Literal["database", "local"]


class _BaseSelectionStrategy(ModynBaseModel):
    maximum_keys_in_memory: int = Field(
        description="Limits how many keys should be materialized at a time in the strategy.",
        ge=1,
    )
    processor_type: str | None = Field(
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

    warmup_triggers: int = Field(
        description=(
            "Defines for how many triggers no pre- or downsampling is performed. "
            "Can be useful to warm up a model on the full dataset before starting to select data."
            "Note that the downsampling schedule is not affected by this, i.e., the schedule "
            "is executed as expected _after_ the warmup period."
        ),
        default=0,
    )

    presampling_config: PresamplingConfig = Field(
        PresamplingConfig(strategy="No", ratio=100),
        description=("Config for the presampling strategy. If missing, no presampling is applied."),
    )
    downsampling_config: SingleDownsamplingConfig | MultiDownsamplingConfig = Field(
        NoDownsamplingConfig(strategy="No", ratio=100),
        description="Configurates the downsampling with one or multiple strategies.",
    )


SelectionStrategy = Annotated[
    FreshnessSamplingStrategyConfig | NewDataStrategyConfig | CoresetStrategyConfig, Field(discriminator="name")
]
