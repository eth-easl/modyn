from collections.abc import Iterable

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
                active_model_trained_before=interval_start,
            )
            for interval_start, interval_end in training_intervals
        ]
