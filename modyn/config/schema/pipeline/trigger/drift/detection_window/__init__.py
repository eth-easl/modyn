from typing import Annotated, Union

from modyn.config.schema.pipeline.trigger.drift.detection_window.window import (
    AmountWindowingStrategy,
    TimeWindowingStrategy,
)
from pydantic import Field

from .window import *  # noqa

DriftWindowingStrategy = Annotated[
    Union[
        AmountWindowingStrategy,
        TimeWindowingStrategy,
    ],
    Field(discriminator="id"),
]
