from __future__ import annotations

from collections.abc import Callable

from pydantic import Field, field_validator

from modyn.config.schema.base_model import ModynBaseModel
from modyn.utils.utils import deserialize_function


class DataConfig(ModynBaseModel):
    dataset_id: str = Field(description="ID of dataset to be used.")
    bytes_parser_function: str = Field(
        description=(
            "Function used to convert bytes received from the Storage, to a format useful for further transformations "
            "(e.g. Tensors) This function is called before any other transformations are performed on the data."
        )
    )
    transformations: list[str] = Field(
        default_factory=list,
        description=(
            "Further transformations to be applied on the data after bytes_parser_function has been applied."
            "For example, this can be torchvision transformations."
        ),
    )
    label_transformer_function: str = Field(
        "", description="Function used to transform the label (tensors of integers)."
    )
    tokenizer: str | None = Field(
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
