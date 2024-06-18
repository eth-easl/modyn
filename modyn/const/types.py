from enum import Enum
from typing import Literal

TimeUnit = Literal["s", "m", "h", "d", "w", "y"]


# typer doesn't support Literal types yet
class TimeResolution(str, Enum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
