from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from modyn.config.schema.base_model import ModynBaseModel
from modyn.utils.utils import EVALUATION_TRANSFORMER_FUNC_NAME, deserialize_function


class _BaseMetricConfig(ModynBaseModel):
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
            deserialize_function(value, EVALUATION_TRANSFORMER_FUNC_NAME)
        except AttributeError as exc:
            raise ValueError("Function 'evaluation_transformer_function' could not be parsed!") from exc
        return value

    @property
    def evaluation_transformer_function_deserialized(self) -> Callable | None:
        if self.evaluation_transformer_function:
            return deserialize_function(self.evaluation_transformer_function, "evaluation_transformer_function")
        return None

    @property
    def shape_check(self) -> bool:
        return True

    @property
    def full_name(self) -> str:
        return self.name


class AccuracyMetricConfig(_BaseMetricConfig):
    name: Literal["Accuracy"] = Field("Accuracy")
    topn: int = Field(1, description="The top-n accuracy to be computed.", ge=1)

    @property
    def shape_check(self) -> bool:
        return self.topn <= 1

    @property
    def full_name(self) -> str:
        return "Accuracy" if self.topn == 1 else f"Top-{self.topn}-Accuracy"


F1ScoreTypes = Literal["macro", "micro", "weighted", "binary"]


class F1ScoreMetricConfig(_BaseMetricConfig):
    name: Literal["F1Score"] = Field("F1Score")
    num_classes: int = Field(description="The total number of classes.")
    average: Literal["macro", "micro", "weighted", "binary"] = Field(
        "macro", description="The method used to average f1-score in the multiclass setting."
    )
    pos_label: int = Field(1, description="The positive label used in binary classification.")

    @model_validator(mode="after")
    def validate_num_classes(self) -> F1ScoreMetricConfig:
        if self.average == "binary" and self.num_classes != 2:
            raise ValueError("Must only have 2 classes for binary F1-score.")
        return self


class RocAucMetricConfig(_BaseMetricConfig):
    name: Literal["RocAuc"] = Field("RocAuc")


class PerplexityMetricConfig(_BaseMetricConfig):
    name: Literal["Perplexity"] = Field("Perplexity")


MetricConfig = Annotated[AccuracyMetricConfig | F1ScoreMetricConfig | RocAucMetricConfig, Field(discriminator="name")]


class _MetricWrapper(BaseModel):
    metric: MetricConfig
    """Only used for validating the internal union type MetricConfig."""


def validate_metric_config_json(metric_config: str) -> MetricConfig:
    """Wrapper for the pydantic model validation (as Union types don't inherit
    from BaseModel)."""
    return _MetricWrapper.model_validate_json('{"metric": ' + str(metric_config) + "}").metric
