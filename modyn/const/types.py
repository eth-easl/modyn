from enum import Enum
from typing import Literal

TimeUnit = Literal["s", "m", "h", "d", "w", "y"]


# typer doesn't support Literal types yet
class TimeResolution(str, Enum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    WEEK = "week"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"


TriggerEvaluationMode = Literal["hindsight", "lookahead"]
"""
Some triggers like PerformanceTrigger and CostTrigger can be evaluated in two different modes:
- hindsight: The trigger is evaluated on the data that has already been seen. Once a we are certain about a threshold
    crossing, the trigger is executed. Triggers are only performed after the threshold was crossed.
- lookahead: The same as hindsight, but we extend the trigger metric with a forecast. The forecast is used to estimate
    the future performance in the next interval. Assuming the forecasted development, we estimate either the time of the next
    trigger or whether the threshold is crossed until the next intervals.
    If that happens before the next update interval, we execute the trigger.
"""

ForecastingMethod = Literal["rolling_average", "ridge_regression"]
"""The approach used for generating the forecasted performance and data density
estimates.

- rolling_average: The forecast is the average of the last n evaluations.
- ridge_regression: The forecast is generated using a ridge regression model.

Note: these options only apply to the `TriggerEvaluationMode="lookahead"` case.
"""
