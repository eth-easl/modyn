from collections.abc import Iterable

from modyn.config.schema.pipeline import StaticEvalStrategyConfig

from .abstract import AbstractEvalStrategy, EvalInterval


class StaticEvalStrategy(AbstractEvalStrategy):
    def __init__(self, config: StaticEvalStrategyConfig):
        super().__init__(config)

    def get_eval_intervals(
        self,
        training_intervals: list[tuple[int, int]],
        dataset_end_time: int | None = None,
    ) -> Iterable[EvalInterval]:
        return [
            EvalInterval(
                start=interval_start,
                end=interval_end,
                # This is independent of the training interval, we use the interval centers as training intervals
                active_model_trained_before=interval_start,
            )
            for interval_start, interval_end in self.config.intervals
        ]
