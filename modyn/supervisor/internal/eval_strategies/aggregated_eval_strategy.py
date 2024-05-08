from typing import Iterable

from modyn.supervisor.internal.eval_strategies.abstract_eval_strategy import AbstractEvalStrategy
from modyn.utils import convert_timestr_to_seconds


class AggregatedEvalStrategy(AbstractEvalStrategy):
    def __init__(self, eval_strategy_config: dict):
        super().__init__(eval_strategy_config)
        self.offsets = eval_strategy_config["offsets"]

    def get_eval_interval(self, first_timestamp: int, last_timestamp: int) -> Iterable[tuple[int, int]]:
        for offset in self.offsets:
            if offset == "-inf":
                yield 0, first_timestamp
            elif offset == "inf":
                # +1 because the left bound of the evaluation interval is inclusive
                yield last_timestamp + 1, -1
            else:
                offset = convert_timestr_to_seconds(offset)
                if offset < 0:
                    yield min(first_timestamp + offset, 0), first_timestamp
                elif offset > 0:
                    yield last_timestamp + 1, last_timestamp + offset + 1
                else:
                    # offset == 0. +1 because the right bound of the evaluation interval is exclusive,
                    # we want to include samples with `last_timestamp` as its timestamp in evaluation dataset
                    yield first_timestamp, last_timestamp + 1
