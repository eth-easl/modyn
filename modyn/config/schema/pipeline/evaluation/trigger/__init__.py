from typing import Annotated, Union

from pydantic import Field

from .after_training import *  # noqa
from .after_training import AfterTrainingEvalTriggerConfig
from .base import *  # noqa
from .periodic import *  # noqa
from .periodic import PeriodicEvalTriggerConfig
from .static import *  # noqa
from .static import StaticEvalTriggerConfig

EvalTriggerConfig = Annotated[
    Union[StaticEvalTriggerConfig, PeriodicEvalTriggerConfig, AfterTrainingEvalTriggerConfig],
    Field(discriminator="mode"),
]
