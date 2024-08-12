from typing import Annotated

from pydantic import Field

from .data_amount import DataAmountTriggerConfig  # noqa
from .time import TimeTriggerConfig  # noqa

SimpleTriggerConfig = Annotated[
    TimeTriggerConfig | DataAmountTriggerConfig,
    Field(discriminator="id"),
]
