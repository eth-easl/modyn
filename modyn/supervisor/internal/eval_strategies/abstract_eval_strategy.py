from abc import ABC, abstractmethod
from typing import Iterable, Optional


class AbstractEvalStrategy(ABC):
    """
    This class represents an abstract evaluation strategy and its subclasses represent concrete evaluation strategies.
    An evaluation strategy determines on which slices of the evaluation dataset the model produced in a trigger is
    evaluated on.

    Given the timespan, i.e. the first and last timestamps of the samples in a trigger, an evaluation strategy returns
    an iterable of tuples via its `get_eval_intervals` method. Each tuple represents an evaluation interval to slice the
    evaluation dataset.
    """
    def __init__(self, eval_strategy_config: dict):
        self.eval_strategy_config = eval_strategy_config

    @abstractmethod
    def get_eval_intervals(
        self, first_timestamp: int, last_timestamp: int
    ) -> Iterable[tuple[Optional[int], Optional[int]]]:
        """
        This method should return an iterable of tuples, where each tuple represents a left inclusive, right exclusive
        evaluation interval. When any side is None, it means that that side is unbounded.
        :param first_timestamp: the timestamp of the first sample in this trigger.
        :param last_timestamp: the timestamp of the last sample in this trigger.
        :return: an iterable of tuples where each tuple represents a
        left inclusive, right exclusive evaluation interval.
        """
        raise NotImplementedError
