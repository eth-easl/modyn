from typing import Iterable, Optional

from modyn.config.schema.pipeline.evaluation import MatrixEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies.abstract_eval_strategy import AbstractEvalStrategy
from modyn.utils import convert_timestr_to_seconds


class MatrixEvalStrategy(AbstractEvalStrategy):
    """
    The MatrixEvalStrategy class represents an evaluation strategy that divides the evaluation dataset
    ranged from `eval_start_from` to `eval_end_at` into fixed-sized intervals. The size of each interval is determined
    by the `eval_every` parameter.

    In case the range from `eval_start_from` to `eval_end_at` is not divisible by `eval_every`, the last interval will
    be smaller than `eval_every`.
    """

    def __init__(self, config: MatrixEvalStrategyConfig):
        super().__init__(config)
        self.eval_every = convert_timestr_to_seconds(config.eval_every)
        self.eval_start_from = config.eval_start_from
        self.eval_end_at = config.eval_end_at

    def get_eval_intervals(
        self, first_timestamp: int, last_timestamp: int
    ) -> Iterable[tuple[Optional[int], Optional[int]]]:
        previous_split = self.eval_start_from
        while True:
            current_split = min(previous_split + self.eval_every, self.eval_end_at)
            yield previous_split, current_split
            if current_split >= self.eval_end_at:
                break
            previous_split = current_split
