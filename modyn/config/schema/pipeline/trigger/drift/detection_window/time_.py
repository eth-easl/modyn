from __future__ import annotations

from functools import cached_property
from typing import Literal

from modyn.const.regex import REGEX_TIME_UNIT
from modyn.utils.utils import SECONDS_PER_UNIT
from pydantic import Field

from .window import _BaseWindowingStrategy


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
