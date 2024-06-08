from typing import Literal

from pydantic import Field

from .base import MatrixEvalTriggerConfig


class StaticEvalTriggerConfig(MatrixEvalTriggerConfig):
    """A user defined sequence of timestamps at which the evaluation should be performed.

    Note: This strategy will run evaluations after the core pipeline.
    """

    mode: Literal["static"] = Field("static")
    at: set[int] = Field(
        description="List of timestamps or sample indexes at which the evaluation should be performed."
    )
    start_timestamp: int = Field(0, description="The timestamp at which evaluations are started to be checked.")
