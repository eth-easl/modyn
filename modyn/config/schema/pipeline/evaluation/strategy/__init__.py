from typing import Annotated, Union

from pydantic import Field

from .between_two_triggers import *  # noqa
from .between_two_triggers import BetweenTwoTriggersEvalStrategyConfig
from .offset import *  # noqa
from .offset import OffsetEvalStrategyConfig
from .periodic import *  # noqa
from .periodic import PeriodicEvalStrategyConfig
from .slicing import *  # noqa
from .slicing import SlicingEvalStrategyConfig
from .static import *  # noqa
from .static import StaticEvalStrategyConfig

EvalStrategyConfig = Annotated[
    Union[
        SlicingEvalStrategyConfig,
        OffsetEvalStrategyConfig,
        BetweenTwoTriggersEvalStrategyConfig,
        PeriodicEvalStrategyConfig,
        StaticEvalStrategyConfig,
    ],
    Field(discriminator="type"),
]
