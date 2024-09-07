from collections.abc import Iterable

from modyn.config.schema.pipeline import BetweenTwoTriggersEvalStrategyConfig

from .abstract import AbstractEvalStrategy, EvalInterval


class BetweenTwoTriggersEvalStrategy(AbstractEvalStrategy):
    def __init__(self, config: BetweenTwoTriggersEvalStrategyConfig):
        super().__init__(config)

    def get_eval_intervals(
        self,
        training_intervals: list[tuple[int, int]],
        dataset_end_time: int | None = None,
    ) -> Iterable[EvalInterval]:
        if not training_intervals:
            return []

        max_training_end = max(interval[1] for interval in training_intervals)
        usage_intervals = training_intervals + (
            [(max_training_end + 1, dataset_end_time)] if dataset_end_time is not None else []
        )
        return [
            EvalInterval(
                start=interval_start,
                end=interval_end,
                active_model_trained_before=interval_start,
            )
            for interval_start, interval_end in usage_intervals
        ]
