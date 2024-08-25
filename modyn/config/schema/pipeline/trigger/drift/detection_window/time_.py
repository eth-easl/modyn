from __future__ import annotations

from functools import cached_property
from typing import Literal

from pydantic import Field

from modyn.utils.utils import SECONDS_PER_UNIT

from .window import _BaseWindowingStrategy


class TimeWindowingStrategy(_BaseWindowingStrategy):
    id: Literal["TimeWindowingStrategy"] = Field("TimeWindowingStrategy")
    limit_ref: str | None = None
    limit_cur: str | None = None
    limit: str | None = None

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
