from abc import ABC, abstractmethod
from typing import Iterable


class AbstractEvalStrategy(ABC):
    def __init__(self, eval_strategy_config: dict):
        self.eval_strategy_config = eval_strategy_config

    @abstractmethod
    def get_eval_interval(self, first_timestamp: int, last_timestamp: int) -> Iterable[tuple[int, int]]:
        """
        This method should return an iterable of tuples, where each tuple represents a left inclusive, right exclusive
        evaluation interval.
        :param first_timestamp: the timestamp of the first sample in this trigger.
        :param last_timestamp: the timestamp of the last sample in this trigger.
        :return: an iterable of tuples where each tuple represents a left inclusive, right exclusive evaluation interval.
        """
        raise NotImplementedError
