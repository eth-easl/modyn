from functools import cached_property
from typing import Literal

from modyn.const.regex import REGEX_TIME_UNIT
from modyn.utils.utils import SECONDS_PER_UNIT
from pydantic import Field

from ._interval_strategy import _IntervalEvalStrategyConfig


class PeriodicEvalStrategyConfig(_IntervalEvalStrategyConfig):
    """Comparable to TimeTriggerConfig and DataAmountTriggerConfig, but for scheduling evaluation runs."""

    type: Literal["PeriodicEvalStrategy"] = Field("PeriodicEvalStrategy")
    every: str = Field(
        description=("Interval length for the evaluation as an integer followed by a time unit (s, m, h, d, w, y)"),
        pattern=rf"^\d+({REGEX_TIME_UNIT})$",
    )
    start_timestamp: int = Field(description="The timestamp at which evaluations are started to be checked.")
    end_timestamp: int = Field(
        description="The timestamp at which evaluations are stopped to be checked. Needed iff unit is 'epoch'."
    )

    @cached_property
    def every_sec(self) -> int:
        unit = str(self.every)[-1:]
        num = int(str(self.every)[:-1])
        return num * SECONDS_PER_UNIT[unit]
