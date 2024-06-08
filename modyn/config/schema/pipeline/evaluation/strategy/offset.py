from __future__ import annotations

from typing import List, Literal

from modyn.config.schema.base_model import ModynBaseModel
from modyn.supervisor.internal.eval_strategies import OffsetEvalStrategy
from modyn.utils import validate_timestr
from pydantic import Field, field_validator


class OffsetEvalStrategyConfig(ModynBaseModel):
    type: Literal["OffsetEvalStrategy"] = Field("OffsetEvalStrategy")
    offsets: List[str] = Field(
        description=(
            "A list of offsets that define the evaluation intervals. For valid offsets, see the class docstring of "
            "OffsetEvalStrategy."
        ),
        min_length=1,
    )

    @field_validator("offsets")
    @classmethod
    def validate_offsets(cls, value: List[str]) -> List[str]:
        for offset in value:
            if offset not in [OffsetEvalStrategy.NEGATIVE_INFINITY, OffsetEvalStrategy.INFINITY]:
                if not validate_timestr(offset):
                    raise ValueError(f"offset {offset} must be a valid time string")
        return value
