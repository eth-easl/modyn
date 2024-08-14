from __future__ import annotations

from typing import Literal

from pydantic import Field

from .window import _BaseWindowingStrategy


class AmountWindowingStrategy(_BaseWindowingStrategy):
    id: Literal["AmountWindowingStrategy"] = Field("AmountWindowingStrategy")
    amount_ref: int = Field(
        1000,
        description="How many data points should fit in the reference window",
        ge=1,
    )
    amount_cur: int = Field(1000, description="How many data points should fit in the current window", ge=1)
