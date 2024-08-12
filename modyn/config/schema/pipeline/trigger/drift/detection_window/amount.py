from __future__ import annotations

from typing import Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel


class _BaseWindowingStrategy(ModynBaseModel):
    allow_overlap: bool = Field(
        False,
        description=(
            "Whether the windows are allowed to overlap. This is useful for time-based windows."
            "If set to False, the current window will be reset after each trigger."
        ),
    )


class AmountWindowingStrategy(_BaseWindowingStrategy):
    id: Literal["AmountWindowingStrategy"] = Field("AmountWindowingStrategy")
    amount_ref: int = Field(
        1000,
        description="How many data points should fit in the reference window",
        ge=1,
    )
    amount_cur: int = Field(1000, description="How many data points should fit in the current window", ge=1)
