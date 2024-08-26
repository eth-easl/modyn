from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .cost import *  # noqa
from .cost import (
    AvoidableMisclassificationCostTriggerConfig,
    DataIncorporationLatencyCostTriggerConfig,
)
from .drift import *  # noqa
from .drift import DataDriftTriggerConfig
from .ensemble import *  # noqa
from .ensemble import EnsembleTriggerConfig
from .performance import *  # noqa
from .performance import PerformanceTriggerConfig
from .simple import *  # noqa
from .simple import SimpleTriggerConfig

TriggerConfig = Annotated[
    AvoidableMisclassificationCostTriggerConfig
    | DataIncorporationLatencyCostTriggerConfig
    | DataDriftTriggerConfig
    | EnsembleTriggerConfig
    | PerformanceTriggerConfig
    | SimpleTriggerConfig,
    Field(discriminator="id"),
]
