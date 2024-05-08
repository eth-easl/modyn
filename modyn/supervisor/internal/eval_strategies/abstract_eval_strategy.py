from abc import ABC, abstractmethod
from typing import Iterable


class AbstractEvalStrategy(ABC):
    def __init__(self, eval_strategy_config: dict):
        self.eval_strategy_config = eval_strategy_config

    @abstractmethod
    def get_eval_interval(self, first_timestamp: int, last_timestamp: int) -> Iterable[tuple[int, int]]:
        raise NotImplementedError
