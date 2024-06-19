from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from modyn.config.schema.pipeline import EvalStrategyConfig


@dataclass
class EvalInterval:
    start: int
    """The start timestamp of the evaluation interval handed over to the evaluation server."""

    end: int | None
    """The end timestamp of the evaluation interval handed over to the evaluation server."""

    active_model_trained_before: int | None = None
    """When determining the most recent model for this EvalInterval, we choose the newest model (highest model_id)
        that satisfies the constraint model_training_interval_end <= active_model_trained_before."""


class AbstractEvalStrategy(ABC):
    """
    This class represents an abstract evaluation strategy and its subclasses represent concrete evaluation strategies.
    An evaluation strategy determines on which slices of the evaluation dataset the model produced in a trigger is
    evaluated on.

    Given the timespan, i.e. the first and last timestamps of the samples in a trigger, an evaluation strategy returns
    an iterable of tuples via its `get_eval_intervals` method. Each tuple represents an evaluation interval to slice the
    evaluation dataset.
    """

    def __init__(self, config: EvalStrategyConfig):
        self.config = config

    @abstractmethod
    def get_eval_intervals(self, training_intervals: Iterable[tuple[int, int]]) -> Iterable[EvalInterval]:
        """
        This method should return an iterable of tuples, where each tuple represents a left inclusive, right exclusive
        evaluation interval. When any side is None, it means that that side is unbounded.
        :param first_timestamp: the timestamp of the first sample in this trigger.
        :param last_timestamp: the timestamp of the last sample in this trigger.
        :return: an iterable of tuples where each tuple represents a
        left inclusive, right exclusive evaluation interval.
        """
        raise NotImplementedError
