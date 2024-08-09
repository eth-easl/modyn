from functools import cached_property
from typing import Literal, cast, get_args

from pydantic import Field, field_validator, model_validator

from modyn.config.schema.base_model import ModynBaseModel
from modyn.const.regex import REGEX_TIME_UNIT
from modyn.const.types import TimeUnit

_EVAL_INTERVAL_BOUND_PATTERN = rf"[+-]?\s*((\d+\s*({REGEX_TIME_UNIT}))|(inf))"
_EVAL_INTERVAL_PATTERN = (
    rf"^(\[|\()\s*({_EVAL_INTERVAL_BOUND_PATTERN})\s*(,|;)\s*({_EVAL_INTERVAL_BOUND_PATTERN})\s*(\)|\])$"
)

IntervalEvalStrategyBounds = Literal["inf", "-inf"]


class _IntervalEvalStrategyConfig(ModynBaseModel):
    """Allows to evaluate a model on an interval that is centered around the
    time of the evaluation trigger. Bounds of the evaluation interval can be
    have an offset relative to the training interval bounds.

    Checkout `docs/EVALUATION.md` for a graphical representation of the options.

    Note: when specifying the left and right offsets you can choose between several
        time units: 's', 'm', 'h', 'd', 'w', 'y'

    Note: for the following we define `training_interval` as the interval that is used for training the model.

    The position of the offset (bounds inclusive):
    - numbers > 0: the boundary of the interval is the end_training_interval + offset
    - numbers < 0: the boundary of the interval is the start_training_interval - offset
    - 'inf': the boundary of the interval is the dataset end
    - '-inf': the boundary of the interval is the dataset start
    - '0', '-0', '+0': zero refers to the training_interval.
        - "-0" = start of the training_interval (start_training_interval)
        - "+0" = end of the training_interval (end_training_interval)
        - '0': start_training_interval if used as left offset, end_training_interval if used as right offset
    """

    interval: str = Field(
        description="The two-sided interval specified by two offsets relative to the training interval.",
        pattern=_EVAL_INTERVAL_PATTERN,
    )
    """e.g. [-2d; +0y], [-inf; +inf]=(-inf; +inf)"""

    @cached_property
    def left(self) -> str:
        return self._parsed[0]

    @cached_property
    def left_unit(self) -> TimeUnit:
        return self._parsed[1]

    @cached_property
    def left_bound_inclusive(self) -> bool:
        return self._parsed[2]

    @cached_property
    def right(self) -> str:
        return self._parsed[3]

    @cached_property
    def right_unit(self) -> TimeUnit:
        return self._parsed[4]

    @cached_property
    def right_bound_inclusive(self) -> bool:
        return self._parsed[5]

    @field_validator("interval")
    @classmethod
    def clean_interval(cls, value: str) -> str:
        return value.strip().replace(" ", "").replace(",", ";")

    @staticmethod
    def _bounds_type(offset: str) -> int:  # pylint: disable=too-many-return-statements
        """Assign temporal sort keys for the different bounds."""
        if offset == "-inf":
            return -3
        if offset == "-0":
            return -1
        if offset == "0":
            return 0
        if offset.startswith("-"):
            return -2
        if offset == "+0":
            return 1
        if offset == "+inf":
            return 3
        return 2

    @model_validator(mode="after")
    def check_offsets(self) -> "_IntervalEvalStrategyConfig":
        # sequential list of offsets: -inf, -number, -0, 0, +0, +number, +inf
        if _IntervalEvalStrategyConfig._bounds_type(self.left) > _IntervalEvalStrategyConfig._bounds_type(self.right):
            raise ValueError("The left offset must be smaller than the right offset.")
        return self

    @cached_property
    def _parsed(self) -> tuple[str, TimeUnit, bool, str, TimeUnit, bool]:
        """Returns:
        the bounds offsets of the interval as seconds
        0: left bound offset
        1: left unit
        2: left_bound_inclusive
        3: right bound offset
        4: right unit
        5: right_bound_inclusive
        """
        interval = str(self.interval)
        left_bound_inclusive = interval[0] == "["
        right_bound_inclusive = interval[-1] == "]"
        interval = interval[1:-1]
        left_raw, right_raw = interval.split(";")
        left_is_inf = "inf" in left_raw
        right_is_inf = "inf" in right_raw

        if left_is_inf:
            assert "+" not in left_raw, "Left bound of the interval cannot be +inf."
            left = "-inf"
            left_unit = "d"
        else:
            left_unit = left_raw[-1]
            left = left_raw[:-1]

        if right_is_inf:
            assert "-" not in right_raw, "Right bound of the interval cannot be -inf."
            right = "+inf"
            right_unit = "d"
        else:
            right_unit = right_raw[-1]
            right = right_raw[:-1]

        assert left_unit in get_args(TimeUnit)
        assert right_unit in get_args(TimeUnit)
        return (
            left,
            cast(TimeUnit, left_unit),
            left_bound_inclusive,
            right,
            cast(TimeUnit, right_unit),
            right_bound_inclusive,
        )
