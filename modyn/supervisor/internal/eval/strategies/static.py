from typing import Iterable

from modyn.config.schema.pipeline import StaticEvalStrategyConfig

from .abstract import AbstractEvalStrategy, EvalInterval


class StaticEvalStrategy(AbstractEvalStrategy):
    def __init__(self, config: StaticEvalStrategyConfig):
        super().__init__(config)

    def get_eval_intervals(self, training_intervals: Iterable[tuple[int, int]]) -> Iterable[EvalInterval]:
        return [
            EvalInterval(
                start=interval_start,
                end=interval_end,
                # This is independent of the training interval, we use the interval centers as training intervals
                training_interval_start=(
                    (interval_end + interval_start) // 2 if interval_end is not None else interval_start
                ),
                training_interval_end=(
                    (interval_end + interval_start) // 2 if interval_end is not None else interval_start
                ),
            )
            for interval_start, interval_end in self.config.intervals
        ]
