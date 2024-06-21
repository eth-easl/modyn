from functools import cached_property
from typing import Annotated, Any, Literal, Union

from modyn.config.schema.base_model import ModynBaseModel
from modyn.const.regex import REGEX_TIME_UNIT
from modyn.utils.utils import SECONDS_PER_UNIT
from pydantic import Field


class TimeTriggerConfig(ModynBaseModel):
    id: Literal["TimeTrigger"] = Field("TimeTrigger")
    every: str = Field(
        description="Interval length for the trigger as an integer followed by a time unit: s, m, h, d, w, y",
        pattern=rf"^\d+{REGEX_TIME_UNIT}$",
    )
    start_timestamp: int | None = Field(
        None,
        description=(
            "The timestamp at which the triggering schedule starts. First trigger will be at start_timestamp + every."
            "Use None to start at the first timestamp of the data."
        ),
    )

    @cached_property
    def every_seconds(self) -> int:
        unit = str(self.every)[-1:]
        num = int(str(self.every)[:-1])
        return num * SECONDS_PER_UNIT[unit]


class DataAmountTriggerConfig(ModynBaseModel):
    id: Literal["DataAmountTrigger"] = Field("DataAmountTrigger")
    num_samples: int = Field(description="The number of samples that should trigger the pipeline.", ge=1)


class DataDriftTriggerConfig(ModynBaseModel):
    id: Literal["DataDriftTrigger"] = Field("DataDriftTrigger")
    detection_interval_data_points: int = Field(
        1000, description="The number of samples in the interval after which drift detection is performed.", ge=1
    )
    sample_size: int | None = Field(None, description="The number of samples used for the metric calculation.", ge=1)
    metric: str = Field("model", description="The metric used for drift detection.")
    metric_config: dict[str, Any] = Field(default_factory=dict, description="Configuration for the evidently metric.")


TriggerConfig = Annotated[
    Union[TimeTriggerConfig, DataAmountTriggerConfig, DataDriftTriggerConfig], Field(discriminator="id")
]
