from __future__ import annotations

from typing import List, Literal

from modyn.config.schema.base_model import ModynBaseModel
from modyn.supervisor.internal.eval_strategies import OffsetEvalStrategy
from modyn.utils import validate_timestr
from pydantic import Field, field_validator


class OffsetEvalStrategyConfig(ModynBaseModel):
    """This evaluation strategy will evaluate the model on an interval around the training interval.

    One of the evaluation interval bounds can have an offset relative to the training interval bounds.

    This strategy can be executed after training, as we know `start_timestamp` and `end_timestamp` of the training
    interval there. The model=matrix option does not make sense in this configuration though as we don't know the
    future `start/end_timestamps` just yet.

    In the `EvalHandlerConfig.execution_time=after_training` case we don't need to restrict the matrix option anymore.
    """

    type: Literal["OffsetEvalStrategy"] = Field("OffsetEvalStrategy")
    offsets: List[str] = Field(
        description=(
            "A list of offsets that define the evaluation intervals. For valid offsets, see the class docstring of "
            "OffsetEvalStrategy."
        ),
        min_length=1,
    )

    @field_validator("offsets")
    @classmethod
    def validate_offsets(cls, value: List[str]) -> List[str]:
        for offset in value:
            if offset not in [OffsetEvalStrategy.NEGATIVE_INFINITY, OffsetEvalStrategy.INFINITY]:
                if not validate_timestr(offset):
                    raise ValueError(f"offset {offset} must be a valid time string")
        return value
