from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from .data_amount import *  # noqa
from .data_amount import DataAmountTriggerConfig
from .drift import *  # noqa
from .drift import DataDriftTriggerConfig
from .ensemble import *  # noqa
from .ensemble import EnsembleTriggerConfig
from .time import *  # noqa
from .time import TimeTriggerConfig

TriggerConfig = Annotated[
    Union[
        TimeTriggerConfig,
        DataAmountTriggerConfig,
        DataDriftTriggerConfig,
        EnsembleTriggerConfig,
    ],
    Field(discriminator="id"),
]
