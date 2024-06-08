from typing import Annotated, Union

from pydantic import Field

from .interval_strategy import *  # noqa
from .interval_strategy import IntervalEvalStrategyConfig
from .matrix_strategy import *  # noqa
from .matrix_strategy import MatrixEvalStrategyConfig
from .offset_strategy import *  # noqa
from .offset_strategy import OffsetEvalStrategyConfig
from .until_next_trigger import *  # noqa
from .until_next_trigger import UntilNextTriggerEvalStrategyConfig

EvalStrategyConfig = Annotated[
    Union[
        MatrixEvalStrategyConfig,
        OffsetEvalStrategyConfig,
        IntervalEvalStrategyConfig,
        UntilNextTriggerEvalStrategyConfig,
    ],
    Field(discriminator="type"),
]
