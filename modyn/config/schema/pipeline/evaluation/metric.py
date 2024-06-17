from __future__ import annotations

from typing import Any, Callable

from modyn.config.schema.base_model import ModynBaseModel
from modyn.utils.utils import deserialize_function
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self


class Metric(ModynBaseModel):
    name: str = Field(description="The name of the evaluation metric.")
    config: dict[str, Any] = Field({}, description="Configuration for the evaluation metric.")
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
            return deserialize_function(self.evaluation_transformer_function or "", "evaluation_transformer_function")
        return None

    @model_validator(mode="after")
    def can_instantiate_metric(self) -> Self:
        # We have to import the MetricFactory here to avoid issues with the multiprocessing context
        # If we move it up, then we'll have `spawn` everywhere, and then the unit tests on Github
        # are way too slow.
        # pylint: disable-next=wrong-import-position,import-outside-toplevel
        from modyn.evaluator.internal.metric_factory import MetricFactory  # fmt: skip  # noqa  # isort:skip
        try:
            MetricFactory.get_evaluation_metric(self.name, self.evaluation_transformer_function or "", self.config)
        except NotImplementedError as exc:
            raise ValueError(f"Cannot instantiate metric {self.name}!") from exc

        return self
