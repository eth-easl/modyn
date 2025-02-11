from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from modyn.config.schema.base_model import ModynBaseModel

OptimizerSource = Literal["PyTorch", "APEX", "HuggingFace"]


class OptimizerParamGroup(ModynBaseModel):
    """Configuration for a parameter group."""

    module: str = Field(description="A set of parameters.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional configuration for the parameter group. (e.g. learning rate)",
    )


class OptimizerConfig(ModynBaseModel):
    """Configuration for the optimizer used during training."""

    name: str = Field(description="The name of the optimizer (like an ID).")
    algorithm: str = Field(description="The type of the optimizer (e.g. SGD).")
    source: OptimizerSource = Field(description="The framework/package the optimizer comes from.")
    param_groups: list[OptimizerParamGroup] = Field(
        description="An array of the parameter groups (parameters and optional configs) this optimizer supervises.",
        min_length=1,
    )


class OptimizationCriterion(ModynBaseModel):
    """Configuration for the optimization criterion that we optimize."""

    name: str = Field(description="The name of the criterion that the pipeline uses (e.g., CrossEntropyLoss).")
    config: dict[str, Any] = Field(default_factory=dict, description="Optional configuration of the criterion.")


LrSchedulerSource = Literal["PyTorch", "Custom"]


class LrSchedulerConfig(ModynBaseModel):
    """Configuration for the Torch-based Learning Rate (LR) scheduler used for
    training."""

    name: str = Field(description="The name of the LR scheduler.")
    source: LrSchedulerSource = Field(description="Source of the LR scheduler.")
    step_every: Literal["epoch", "batch"] = Field(description="Whether to call scheduler.step() every batch or epoch.")
    optimizers: list[str] = Field(
        description="List of optimizers that this scheduler is responsible for.",
        min_length=1,
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional configuration of the lr scheduler. Passed to the lr scheduler as a dict."
            "You can use MODYN_NUM_BATCHES and MODYN_NUM_EPOCHS as string literal and "
            "Modyn will replace it with the number of epochs/batches we will train the model on."
        ),
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
    drop_last_batch: bool = Field(
        default=True, description="Whether to drop the last batch if it is smaller than the batch size."
    )
    shuffle: bool = Field(
        description=(
            "If True, we shuffle the order of partitions and the data within each partition at each worker."
            "Otherwise, the output order is deterministic."
        )
    )
    enable_accurate_gpu_measurements: bool = Field(
        default=False,
        description="If True, we measure the time of individual GPU related operations within a training process more "
        "accurately by cuda synchronization. Note this can have a significant impact on performance on training.",
    )
    use_previous_model: bool = Field(
        description=(
            "If True, on trigger, we continue training on the model outputted by the previous trigger. If False, "
            "we start with random weights. If initial_model is 'pretrained', cannot be False."
        )
    )
    generative: bool = Field(
        False,
        description=(
            "If True then, then the training pipeline goes into the generative branch, data is sampled without expecting labels."
        ),
    )
    lora: bool = Field(
        False,
        description=("Applies Lora layers to the model"),
    )
    kadapter: bool = Field(
        False,
        description=("Applies kadapter layers to the model"),
    )

    seed: int | None = Field(
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
    initial_model_id: int | None = Field(
        None,
        description="The ID of the model that should be used as the initial model.",
    )
    checkpointing: CheckpointingConfig = Field(description="Configuration of checkpointing during training")
    record_loss_every: int = Field(
        default=0,
        description="Record the training loss in the trainer_log very n-th batch/step. If 0, loss is not recorded.",
    )
    optimizers: list[OptimizerConfig] = Field(
        description="An array of the optimizers for the training",
        min_length=1,
    )
    optimization_criterion: OptimizationCriterion = Field(
        description="Configuration for the optimization criterion that we optimize",
    )
    lr_scheduler: LrSchedulerConfig | None = Field(
        None,
        description="Configuration for the Torch-based Learning Rate (LR) scheduler used for training.",
    )
    grad_scaler_config: dict[str, Any] | None = Field(
        None,
        description="Configuration for the torch.cuda.amp.GradScaler. Effective only when amp is enabled.",
    )
    grad_norm: int = Field(
        default=0,
        description="Clips the gradients normed over this value, if its 0 it will not be used.",
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


class CheckpointingConfig(ModynBaseModel):
    """Configuration for checkpointing during training."""

    activated: bool = Field(description="Whether we checkpoint or not.")
    interval: int | None = Field(None, description="In what interval we checkpoint.")
    path: Path | None = Field(
        None,
        description="The path on the training node where the checkpoints are stored.",
    )

    @model_validator(mode="after")
    def validate_activation(self) -> Self:
        if self.activated:
            if self.interval is None:
                raise ValueError("If checkpointing is activated, the interval must be set.")
            if self.path is None:
                raise ValueError("If checkpointing is activated, the path must be set.")
        return self
