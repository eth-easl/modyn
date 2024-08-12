from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from .cost import *  # noqa
from .cost import CostTriggerConfig
from .data_amount import *  # noqa
from .data_amount import DataAmountTriggerConfig
from .drift import *  # noqa
from .drift import DataDriftTriggerConfig
from .ensemble import *  # noqa
from .ensemble import EnsembleTriggerConfig
from .performance import *  # noqa
from .performance import PerformanceTriggerConfig
from .time import *  # noqa
from .time import TimeTriggerConfig

TriggerConfig = Annotated[
    Union[
        TimeTriggerConfig,
        DataAmountTriggerConfig,
        DataDriftTriggerConfig,
        EnsembleTriggerConfig,
        PerformanceTriggerConfig,
        CostTriggerConfig,
    ],
    Field(discriminator="id"),
]
