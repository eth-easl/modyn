from collections.abc import Iterable

from modyn.config.schema.pipeline import PeriodicEvalStrategyConfig

from ._interval import _IntervalEvalStrategyMixin
from .abstract import AbstractEvalStrategy, EvalInterval


class PeriodicEvalStrategy(AbstractEvalStrategy, _IntervalEvalStrategyMixin):
    def __init__(self, config: PeriodicEvalStrategyConfig):
        super(AbstractEvalStrategy, self).__init__(config)
        _IntervalEvalStrategyMixin.__init__(self, config)

    def get_eval_intervals(self, training_intervals: Iterable[tuple[int, int]]) -> Iterable[EvalInterval]:
        evaluation_timestamps = list(
            range(self.config.start_timestamp, self.config.end_timestamp + 1, self.config.every_sec)
        )

        eval_intervals = []
        for eval_timestamp in evaluation_timestamps:
            adjusted_start, adjusted_end = self._generate_interval(eval_timestamp, eval_timestamp)
            eval_intervals.append(
                EvalInterval(
                    start=adjusted_start,
                    end=adjusted_end,
                    # evaluations are independent of training intervals and centered around the evaluation timestamp
                    active_model_trained_before=eval_timestamp,
                )
            )

        return eval_intervals
