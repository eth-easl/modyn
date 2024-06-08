from __future__ import annotations

from typing import Optional

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field

from .data import DataConfig
from .evaluation.config import EvaluationConfig
from .model import ModelConfig
from .model_storage import PipelineModelStorageConfig
from .sampling.config import SelectionStrategy
from .training import TrainingConfig
from .trigger import TriggerConfig


class Pipeline(ModynBaseModel):
    name: str = Field(description="The name of the pipeline.")
    description: Optional[str] = Field(None, description="The description of the pipeline.")
    version: Optional[str] = Field(None, description="The version of the pipeline.")


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
