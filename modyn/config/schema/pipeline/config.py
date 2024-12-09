from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from modyn.config.schema.base_model import ModynBaseModel

from .data import DataConfig
from .evaluation.config import EvaluationConfig
from .model import ModelConfig
from .model_storage import PipelineModelStorageConfig
from .sampling.config import CoresetStrategyConfig, SelectionStrategy
from .sampling.downsampling_config import MultiDownsamplingConfig
from .training import TrainingConfig
from .trigger import TriggerConfig


class Pipeline(ModynBaseModel):
    name: str = Field(description="The name of the pipeline.")
    description: str | None = Field(None, description="The description of the pipeline.")
    version: str | None = Field(None, description="The version of the pipeline.")


class ModynPipelineConfig(ModynBaseModel):
    pipeline: Pipeline

    # model is a reserved keyword in Pydantic, so we use modyn_model instead
    modyn_model: ModelConfig = Field(alias="model")
    modyn_model_storage: PipelineModelStorageConfig = Field(alias="model_storage")

    training: TrainingConfig
    data: DataConfig
    trigger: TriggerConfig  # type: ignore[valid-type]
    selection_strategy: SelectionStrategy
    evaluation: EvaluationConfig | None = Field(None)

    @model_validator(mode="after")
    def validate_bts_training_selection_works(self) -> Self:
        # Validates that when using Downsampling with BtS, we choose a functional ratio
        if isinstance(self.selection_strategy, CoresetStrategyConfig) and not isinstance(
            self.selection_strategy.downsampling_config, MultiDownsamplingConfig
        ):
            if not self.selection_strategy.downsampling_config.sample_then_batch:  # bts
                ratio = self.selection_strategy.downsampling_config.ratio
                ratio_max = self.selection_strategy.downsampling_config.ratio_max
                batch_size = self.training.batch_size

                post_downsampling_size = max((ratio * batch_size) // ratio_max, 1)
                if batch_size % post_downsampling_size != 0:
                    raise ValueError(
                        f"The target batch size of {batch_size} is not a multiple of the batch size "
                        + f"after downsampling with ratio {ratio} a batch in BtS mode ({post_downsampling_size}). "
                        + "We cannot accumulate batches. "
                        + "Please choose the downsampling ratio and batch size such that this is possible."
                    )

        return self


ModynPipelineConfig.model_rebuild()
