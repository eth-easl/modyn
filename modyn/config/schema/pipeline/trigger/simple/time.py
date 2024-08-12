from functools import cached_property
from typing import Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel
from modyn.const.regex import REGEX_TIME_UNIT
from modyn.utils.utils import SECONDS_PER_UNIT


class TimeTriggerConfig(ModynBaseModel):
    id: Literal["TimeTrigger"] = Field("TimeTrigger")
    every: str = Field(
        description="Interval length for the trigger as an integer followed by a time unit: s, m, h, d, w, y",
        pattern=rf"^\d+{REGEX_TIME_UNIT}$",
    )
    start_timestamp: int | None = Field(
        None,
        description=(
            "The timestamp at which the triggering schedule starts. First trigger will be at start_timestamp + every."
            "Use None to start at the first timestamp of the data."
        ),
    )

    @cached_property
    def every_seconds(self) -> int:
        unit = str(self.every)[-1:]
        num = int(str(self.every)[:-1])
        return num * SECONDS_PER_UNIT[unit]
