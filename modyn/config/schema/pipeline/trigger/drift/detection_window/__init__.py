from typing import Annotated

from pydantic import Field

from .amount import AmountWindowingStrategy
from .time_ import TimeWindowingStrategy

DriftWindowingStrategy = Annotated[
    AmountWindowingStrategy | TimeWindowingStrategy,
    Field(discriminator="id"),
]
