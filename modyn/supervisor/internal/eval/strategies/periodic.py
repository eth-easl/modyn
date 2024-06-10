from typing import Iterable, Optional

from modyn.config.schema.pipeline import PeriodicEvalStrategyConfig

from .abstract import AbstractEvalStrategy


class PeriodicEvalStrategy(AbstractEvalStrategy):
    def __init__(self, config: PeriodicEvalStrategyConfig):
        super().__init__(config)

    def get_eval_intervals(
        self, first_timestamp: int, last_timestamp: int
    ) -> Iterable[tuple[Optional[int], Optional[int]]]:
        raise NotImplementedError()
