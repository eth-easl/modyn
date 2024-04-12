from typing import Iterable

from modyn.supervisor.internal.eval_strategies.abstract_eval_strategy import AbstractEvalStrategy
from modyn.utils import convert_timestr_to_seconds


class CustomEvalStrategy(AbstractEvalStrategy):
    def __init__(self, eval_strategy_config: dict):
        super().__init__(eval_strategy_config)
        self.granularity = convert_timestr_to_seconds(self.eval_strategy_config["granularity"])
        assert self.granularity > 0, "granularity must be greater than 0"

    def get_eval_interval(self, first_timestamp: int, last_timestamp: int) -> Iterable[tuple[int, int]]:
        yield from [
            (0, first_timestamp),  # historical data
            (first_timestamp, last_timestamp + 1),  # current data
            (last_timestamp + 1, last_timestamp + 1 + self.granularity),  # future data
        ]
