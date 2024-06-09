
from typing import Iterable, Optional

from modyn.config.schema.pipeline import StaticEvalTriggerConfig

from .abstract import AbstractEvalStrategy


class StaticEvalTrigger(AbstractEvalStrategy):
    def __init__(self, config: StaticEvalTriggerConfig):
        super().__init__(config)

    def get_eval_intervals(
        self, first_timestamp: int, last_timestamp: int
    ) -> Iterable[tuple[Optional[int], Optional[int]]]:
        raise NotImplementedError()
