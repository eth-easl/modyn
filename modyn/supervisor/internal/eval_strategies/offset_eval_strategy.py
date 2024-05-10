from typing import Iterable, Optional

from modyn.supervisor.internal.eval_strategies.abstract_eval_strategy import AbstractEvalStrategy
from modyn.utils import convert_timestr_to_seconds


class OffsetEvalStrategy(AbstractEvalStrategy):
    """
    This evaluation strategy will evaluate the model on the intervals defined by the user provided offsets.
    The offsets are defined as a list of strings, where each string represents an offset.
    An offset can be one of the following:
    - "-inf": the evaluation interval is from the beginning of evaluation dataset to just before this trigger.
    - "inf": the evaluation interval is from just after this trigger to the end of evaluation dataset.
    - an integer followed by a time unit (e.g. "100s", "-100s"):
        If the represented offset is positive, the evaluation interval starts from just after this trigger with length
         of the offset.
        If the represented offset is negative, the evaluation interval ends just before this trigger with length of the
         offset.
        If the represented offset is zero, the evaluation interval is exactly the interval of the trigger.
    """

    def __init__(self, eval_strategy_config: dict):
        super().__init__(eval_strategy_config)
        self.offsets = eval_strategy_config["offsets"]

    def get_eval_interval(
        self, first_timestamp: int, last_timestamp: int
    ) -> Iterable[tuple[Optional[int], Optional[int]]]:
        for offset in self.offsets:
            if offset == "-inf":
                yield 0, first_timestamp
            elif offset == "inf":
                # +1 because the left bound of the returned interval should inclusive, and last_timestamp is included
                # in the current trigger
                yield last_timestamp + 1, None
            else:
                offset = convert_timestr_to_seconds(offset)
                if offset < 0:
                    yield max(first_timestamp + offset, 0), first_timestamp
                elif offset > 0:
                    yield last_timestamp + 1, last_timestamp + offset + 1
                else:
                    # offset == 0. +1 because the right bound of the evaluation interval is exclusive,
                    # we want to include samples with `last_timestamp` as its timestamp in evaluation dataset
                    yield first_timestamp, last_timestamp + 1
