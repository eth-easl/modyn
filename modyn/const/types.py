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


TriggerEvaluationMode = Literal["hindsight", "lookahead"]
"""
- hindsight: The trigger is evaluated on the data that has already been seen. Once a we are certain about a threshold
    crossing, the trigger is executed.
- lookahead: The same as hindsight, but we extend the trigger metric with a forecast. The forecast is used to estimate
    the future performance. Assuming the forecasted development, we estimate the time of the next trigger. If it's
    before the next update interval, we execute the trigger.
    If a hindsight trigger would have been executed, we execute the trigger without having to forecast the future.
"""
