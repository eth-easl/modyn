from typing import Annotated, Union

from pydantic import Field

from .amount import AmountWindowingStrategy
from .time_ import TimeWindowingStrategy

DriftWindowingStrategy = Annotated[
    Union[
        AmountWindowingStrategy,
        TimeWindowingStrategy,
    ],
    Field(discriminator="id"),
]
