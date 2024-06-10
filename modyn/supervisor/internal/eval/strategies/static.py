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
                most_recent_model_interval_end_before=interval_end or interval_start,
            )
            for interval_start, interval_end in self.config.intervals
        ]
