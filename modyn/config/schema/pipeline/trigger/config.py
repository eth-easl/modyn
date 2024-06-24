"""Define the Union type for all trigger configs while not creating a circular import."""

from typing import TYPE_CHECKING, Annotated, Union

from pydantic import Field

if TYPE_CHECKING:
    # use forward references and conditional imports to avoid circular imports
    from .data_amount import DataAmountTriggerConfig
    from .drift import DataDriftTriggerConfig
    from .time import TimeTriggerConfig

TriggerConfig = Annotated[
    Union[
        "DataAmountTriggerConfig",
        "TimeTriggerConfig",
        "DataDriftTriggerConfig",
    ],
    Field(discriminator="id"),
]
