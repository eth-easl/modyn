from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field, field_validator

from ..data import DataConfig
from .metric import Metric
from .strategy import MatrixEvalStrategyModel, OffsetEvalStrategyModel

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