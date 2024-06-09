from __future__ import annotations

from typing import Iterable, Optional

from modyn.config.schema.pipeline.evaluation.strategy._interval_strategy import _IntervalEvalStrategyConfig
from modyn.utils.utils import SECONDS_PER_UNIT

from .abstract import AbstractEvalStrategy


class _IntervalEvalStrategy(AbstractEvalStrategy):
    """See `_IntervalEvalStrategyConfig` for more information."""

    INFINITY = "inf"
    NEGATIVE_INFINITY = "-inf"

    def __init__(self, config: _IntervalEvalStrategyConfig):
        super().__init__(config)

    def get_eval_intervals(
        self, first_timestamp: int, last_timestamp: int
    ) -> Iterable[tuple[Optional[int], Optional[int]]]:
        left = 0
        right: int | None = 0

        if self.config.left == "-inf":
            left = 0
        elif self.config.left in {"0", "-0"}:
            left = first_timestamp + int(not self.config.left_bound_inclusive)
        elif self.config.left.startswith("-"):
            left = (
                first_timestamp
                + int(self.config.left) * SECONDS_PER_UNIT[self.config.left_unit]
                + int(not self.config.left_bound_inclusive)
            )
        elif self.config.left == "+0":
            left = last_timestamp + int(not self.config.left_bound_inclusive)
        elif self.config.left == "+inf":
            raise ValueError("Left bound of the interval cannot be +inf.")
        else:
            # +number
            left = (
                last_timestamp
                + int(self.config.left) * SECONDS_PER_UNIT[self.config.left_unit]
                + int(not self.config.left_bound_inclusive)
            )

        if self.config.right == "-inf":
            raise ValueError("Right bound of the interval cannot be -inf.")
        if self.config.right == "-0":
            right = first_timestamp - int(not self.config.right_bound_inclusive)
        elif self.config.right.startswith("-"):
            right = (
                first_timestamp
                + int(self.config.right) * SECONDS_PER_UNIT[self.config.right_unit]
                - int(not self.config.right_bound_inclusive)
            )
        elif self.config.right in {"0", "+0"}:
            right = last_timestamp - int(not self.config.right_bound_inclusive)
        elif self.config.right == "+inf":
            right = None
        else:
            # +number
            right = (
                last_timestamp
                + int(self.config.right) * SECONDS_PER_UNIT[self.config.right_unit]
                - int(not self.config.right_bound_inclusive)
            )
        yield max(0, left), max(left, right) if right else right
