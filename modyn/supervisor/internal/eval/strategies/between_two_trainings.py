from typing import Iterable

from modyn.config.schema.pipeline import BetweenTwoTriggersEvalStrategyConfig

from .abstract import AbstractEvalStrategy, EvalInterval


class BetweenTwoTriggersEvalStrategy(AbstractEvalStrategy):
    def __init__(self, config: BetweenTwoTriggersEvalStrategyConfig):
        super().__init__(config)

    def get_eval_intervals(self, training_intervals: Iterable[tuple[int, int]]) -> Iterable[EvalInterval]:
        return [
            EvalInterval(
                start=interval_start,
                end=interval_end,
                # This is independent of the training interval, we use the interval centers as training intervals
                training_interval_start=(interval_end + interval_start) // 2,
                training_interval_end=(interval_end + interval_start) // 2,
            )
            for interval_start, interval_end in training_intervals
        ]
