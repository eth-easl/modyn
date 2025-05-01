from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from modyn.config.schema.base_model import ModynBaseModel
from modyn.config.schema.pipeline.training.config import LrSchedulerConfig, OptimizationCriterion, OptimizerConfig

from ..data import DataConfig
from .handler import EvalHandlerConfig
from .metrics import MetricConfig


class EvalDataConfig(DataConfig):
    batch_size: int = Field(description="The batch size to be used during evaluation.", ge=1)
    dataloader_workers: int = Field(
        description="The number of data loader workers on the evaluation node that fetch data from storage.", ge=1
    )
    metrics: list[MetricConfig] = Field(
        description="All metrics used to evaluate the model on the given dataset.",
        min_length=1,
    )
    generative: bool = Field(False, description="Whether the task is generative.")


class ResultWriter(ModynBaseModel):
    name: str = Field(description="The name of the result writer.")
    config: dict[str, Any] | None = Field(None, description="Optional configuration for the result writer.")


ResultWriterType = Literal["json", "json_dedicated", "tensorboard"]
"""
- json: appends the evaluations to the standard json logfile.
- json_dedicated: dumps the results into dedicated json files for each evaluation.
- tensorboard: output the evaluation to dedicated tensorboard files."""


class TuningConfig(ModynBaseModel):
    epochs: int = Field(1, description="Number of epochs for tuning.", ge=1)
    num_samples_to_pass: list[int] | None = Field(None, description="Number of samples to pass per epoch.")

    batch_size: int = Field(1, description="Batch size used in tuning.", ge=1)
    dataloader_workers: int = Field(1, description="Number of workers for data loading.", ge=1)
    drop_last_batch: bool = Field(True, description="Whether to drop the last batch if smaller than batch size.")
    shuffle: bool = Field(True, description="Whether data is shuffled during tuning.")
    enable_accurate_gpu_measurements: bool = Field(False, description="Enable precise GPU measurement during tuning.")

    amp: bool = Field(False, description="Whether automatic mixed precision is enabled.")
    lr_scheduler: LrSchedulerConfig | None = Field(None, description="Learning rate scheduler configuration.")
    device: str = Field(
        "cpu",
        description="The device the model should be put on.",
        pattern=r"^(cpu|cuda:\d+)$",
    )

    seed: int | None = Field(None, description="Random seed for reproducibility, if provided.")
    optimizers: list[OptimizerConfig] = Field(
        description="An array of the optimizers for the training",
        min_length=1,
    )
    optimization_criterion: OptimizationCriterion = Field(
        description="Configuration for the optimization criterion that we optimize",
    )
    datasets: EvalDataConfig = Field(
        description="Dataset used for light tuning",
        min_length=1,
    )


class EvaluationConfig(ModynBaseModel):
    handlers: list[EvalHandlerConfig] = Field(
        description="An array of all evaluation handlers that should be used to evaluate the model.",
        min_length=1,
    )
    device: str = Field(description="The device the model should be put on.")
    datasets: list[EvalDataConfig] = Field(
        description="An array of all datasets on which the model is evaluated.",
        min_length=1,
    )

    after_training_evaluation_workers: int = Field(5, description="Number of workers for after training evaluation.")
    after_pipeline_evaluation_workers: int = Field(5, description="Number of workers for post pipeline evaluation.")

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: list[EvalDataConfig]) -> list[EvalDataConfig]:
        dataset_ids = [dataset.dataset_id for dataset in value]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Dataset IDs must be unique.")
        return value
