from __future__ import annotations

from functools import cached_property
from typing import Annotated, Literal, Union

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel
from modyn.const.regex import REGEX_TIME_UNIT
from modyn.utils.utils import SECONDS_PER_UNIT


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
    amount_ref: int = Field(1000, description="How many data points should fit in the reference window", ge=1)
    amount_cur: int = Field(1000, description="How many data points should fit in the current window", ge=1)

    @property
    def current_buffer_size(self) -> int | None:
        return self.amount_cur

    @property
    def reference_buffer_size(self) -> int | None:
        return self.amount_ref


class TimeWindowingStrategy(_BaseWindowingStrategy):
    id: Literal["TimeWindowingStrategy"] = Field("TimeWindowingStrategy")
    limit_ref: str = Field(
        description="Window size as an integer followed by a time unit: s, m, h, d, w, y",
        pattern=rf"^\d+{REGEX_TIME_UNIT}$",
    )
    limit_cur: str = Field(
        description="Window size as an integer followed by a time unit: s, m, h, d, w, y",
        pattern=rf"^\d+{REGEX_TIME_UNIT}$",
    )

    @cached_property
    def limit_seconds_ref(self) -> int:
        unit = str(self.limit_ref)[-1:]
        num = int(str(self.limit_ref)[:-1])
        return num * SECONDS_PER_UNIT[unit]

    @cached_property
    def limit_seconds_cur(self) -> int:
        unit = str(self.limit_cur)[-1:]
        num = int(str(self.limit_cur)[:-1])
        return num * SECONDS_PER_UNIT[unit]

    @property
    def current_buffer_size(self) -> int | None:
        return None

    @property
    def reference_buffer_size(self) -> int | None:
        return None


DriftWindowingStrategy = Annotated[
    Union[
        AmountWindowingStrategy,
        TimeWindowingStrategy,
    ],
    Field(discriminator="id"),
]
