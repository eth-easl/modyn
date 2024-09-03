from collections.abc import Iterable

from modyn.config.schema.pipeline import SlicingEvalStrategyConfig

from .abstract import AbstractEvalStrategy, EvalInterval


class SlicingEvalStrategy(AbstractEvalStrategy):
    """The SlicingEvalStrategy class represents an evaluation strategy that
    divides the evaluation dataset ranged from `eval_start_from` to
    `eval_end_at` into fixed-sized intervals. The size of each interval is
    determined by the `eval_every` parameter.

    In case the range from `eval_start_from` to `eval_end_at` is not divisible by `eval_every`, the last interval will
    be smaller than `eval_every`.
    """

    def __init__(self, config: SlicingEvalStrategyConfig):
        super().__init__(config)

    def get_eval_intervals(
        self,
        training_intervals: list[tuple[int, int]],
        dataset_end_time: int | None = None,
    ) -> Iterable[EvalInterval]:
        previous_split = self.config.eval_start_from
        while True:
            current_split = min(previous_split + self.config.eval_every_sec, self.config.eval_end_at)
            yield EvalInterval(
                start=previous_split,
                end=current_split,
                active_model_trained_before=previous_split,
            )
            if current_split >= self.config.eval_end_at:
                break
            previous_split = current_split
