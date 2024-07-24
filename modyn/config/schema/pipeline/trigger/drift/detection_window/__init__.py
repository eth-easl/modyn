from typing import Annotated, Union

from pydantic import Field

from modyn.config.schema.pipeline.trigger.drift.detection_window.window import (
    AmountWindowingStrategy,
    TimeWindowingStrategy,
)

from .window import *  # noqa

DriftWindowingStrategy = Annotated[
    Union[
        AmountWindowingStrategy,
        TimeWindowingStrategy,
    ],
    Field(discriminator="id"),
]
