from typing import Annotated, Union

from pydantic import Field

from .offset import *  # noqa
from .offset import OffsetEvalStrategyConfig
from .slicing import *  # noqa
from .slicing import SlicingEvalStrategyConfig

EvalStrategyConfig = Annotated[
    Union[
        SlicingEvalStrategyConfig,
        OffsetEvalStrategyConfig,
    ],
    Field(discriminator="type"),
]
