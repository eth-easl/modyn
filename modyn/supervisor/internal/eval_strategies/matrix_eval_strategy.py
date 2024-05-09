from __future__ import annotations

from typing import Iterable

from modyn.supervisor.internal.eval_strategies.abstract_eval_strategy import AbstractEvalStrategy
from modyn.utils import convert_timestr_to_seconds


class MatrixEvalStrategy(AbstractEvalStrategy):
    def __init__(self, eval_strategy_config: dict):
        super().__init__(eval_strategy_config)
        self.eval_every = convert_timestr_to_seconds(self.eval_strategy_config["eval_every"])
        self.eval_start_from = self.eval_strategy_config["eval_start_from"]
        self.eval_end_at = self.eval_strategy_config["eval_end_at"]
        assert self.eval_start_from < self.eval_end_at, "eval_start_from must be less than eval_end_at"
        assert self.eval_every > 0, "eval_every must be greater than 0"

    def get_eval_interval(self, first_timestamp: int, last_timestamp: int) -> Iterable[tuple[int | None, int | None]]:
        previous_split = self.eval_start_from
        while True:
            current_split = min(previous_split + self.eval_every, self.eval_end_at)
            yield previous_split, current_split
            if current_split == self.eval_end_at:
                break
            previous_split = current_split
