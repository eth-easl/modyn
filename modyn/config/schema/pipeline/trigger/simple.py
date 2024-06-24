"""Module providing a collection of simple trigger policies that can be used
to configure the evaluation interval of other, more complex triggers.

We don't support using complex trigger policies to configure other complex
trigger policies, as this would yield in cyclic dependencies and does not make much
sense in practice.
"""


from typing import Annotated, Union

from pydantic import Field

from .data_amount import DataAmountTriggerConfig
from .time import TimeTriggerConfig

SimpleTriggerConfig = Annotated[
    Union[
        DataAmountTriggerConfig,
        TimeTriggerConfig,
    ],
    Field(discriminator="id"),
]
