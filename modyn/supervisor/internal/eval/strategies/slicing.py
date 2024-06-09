from typing import Iterable, Optional

from modyn.config.schema.pipeline import SlicingEvalStrategyConfig

from .abstract import AbstractEvalStrategy


class SlicingEvalStrategy(AbstractEvalStrategy):
    """
    The SlicingEvalStrategy class represents an evaluation strategy that divides the evaluation dataset
    ranged from `eval_start_from` to `eval_end_at` into fixed-sized intervals. The size of each interval is determined
    by the `eval_every` parameter.

    In case the range from `eval_start_from` to `eval_end_at` is not divisible by `eval_every`, the last interval will
    be smaller than `eval_every`.
    """

    def __init__(self, config: SlicingEvalStrategyConfig):
        super().__init__(config)

    def get_eval_intervals(
        self, first_timestamp: int, last_timestamp: int
    ) -> Iterable[tuple[Optional[int], Optional[int]]]:
        previous_split = self.config.eval_start_from
        while True:
            current_split = min(previous_split + self.config.eval_every_sec, self.config.eval_end_at)
            yield previous_split, current_split
            if current_split >= self.config.eval_end_at:
                break
            previous_split = current_split
