from collections.abc import Iterable

from modyn.config.schema.pipeline import OffsetEvalStrategyConfig
from modyn.utils import convert_timestr_to_seconds

from .abstract import AbstractEvalStrategy, EvalInterval


class OffsetEvalStrategy(AbstractEvalStrategy):
    """This evaluation strategy will evaluate the model on the intervals
    defined by the user provided offsets. The offsets are defined as a list of
    strings, where each string represents an offset.

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

    def __init__(self, config: OffsetEvalStrategyConfig):
        super().__init__(config)
        self.offsets = config.offsets

    def get_eval_intervals(self, training_intervals: Iterable[tuple[int, int]]) -> Iterable[EvalInterval]:
        for train_interval_start, train_interval_end in training_intervals:
            for offset in self.offsets:
                if offset == "-inf":
                    yield EvalInterval(
                        start=0,
                        end=train_interval_start,
                        # strategy only applicable after training where active models cannot be determined
                        active_model_trained_before=None,
                    )
                elif offset == "inf":
                    # +1 because the samples with timestamp `train_interval_end` are included in the current trigger,
                    # and here we want to return an interval on the data with timestamp greater than
                    # `train_interval_end`.
                    yield EvalInterval(
                        start=train_interval_end + 1,
                        end=None,
                        active_model_trained_before=None,
                    )
                else:
                    offset_int = convert_timestr_to_seconds(offset)
                    if offset_int < 0:
                        yield EvalInterval(
                            start=max(train_interval_start + offset_int, 0),
                            end=train_interval_start,
                            active_model_trained_before=None,
                        )
                    elif offset_int > 0:
                        # +1 for the same reason as above
                        yield EvalInterval(
                            start=train_interval_end + 1,
                            end=train_interval_end + offset_int + 1,
                            active_model_trained_before=None,
                        )
                    else:
                        # now offset_int == 0. We want to return the same interval as the trigger's interval.
                        # +1 because the right bound of the returned interval should be exclusive.
                        # we want to include samples with timestamp `train_interval_end` from evaluation dataset.
                        yield EvalInterval(
                            start=train_interval_start,
                            end=train_interval_end + 1,
                            active_model_trained_before=None,
                        )
