from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, NonNegativeInt, field_validator, model_validator

from modyn.config.schema.base_model import ModynBaseModel
from modyn.utils import validate_timestr
from modyn.utils.utils import SECONDS_PER_UNIT


class SlicingEvalStrategyConfig(ModynBaseModel):
    type: Literal["SlicingEvalStrategy"] = Field("SlicingEvalStrategy")
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

    @property
    def eval_every_sec(self) -> int:
        unit = str(self.eval_every)[-1:]
        num = int(str(self.eval_every)[:-1])
        return num * SECONDS_PER_UNIT[unit]
