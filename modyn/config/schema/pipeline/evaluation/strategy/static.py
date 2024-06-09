from typing import Literal

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class StaticEvalTriggerConfig(ModynBaseModel):
    """A sequence of intervals at which the evaluation should be performed."""

    type: Literal["StaticEvalStrategy"] = Field("StaticEvalStrategy")
    intervals: set[tuple[int, int | None]] = Field(
        description=(
            "List of intervals in which the evaluation should be performed."
            "Each interval is a tuple of two timestamps. Use open end."
        )
    )
