from __future__ import annotations

from functools import cached_property
from typing import Annotated, ForwardRef, Literal, Optional, Union

from modyn.config.schema.base_model import ModynBaseModel
from modyn.const.regex import REGEX_TIME_UNIT
from modyn.utils.utils import SECONDS_PER_UNIT
from pydantic import Field

from .aggregation import DriftAggregationStrategy, MajorityVoteDriftAggregationStrategy
from .alibi_detect import AlibiDetectDriftMetric
from .evidently import EvidentlyDriftMetric

__TriggerConfig = ForwardRef("TriggerConfig", is_class=True)

DriftMetric = Annotated[
    Union[
        EvidentlyDriftMetric,
        AlibiDetectDriftMetric,
    ],
    Field(discriminator="id"),
]


class AmountWindowingStrategy(ModynBaseModel):
    id: Literal["AmountWindowingStrategy"] = Field("AmountWindowingStrategy")
    amount: int = Field(1000, description="How many data points should fit in the window")


class TimeWindowingStrategy(ModynBaseModel):
    id: Literal["TimeWindowingStrategy"] = Field("TimeWindowingStrategy")
    limit: str = Field(
        description="Window size as an integer followed by a time unit: s, m, h, d, w, y",
        pattern=rf"^\d+{REGEX_TIME_UNIT}$",
    )

    @cached_property
    def limit_seconds(self) -> int:
        unit = str(self.limit)[-1:]
        num = int(str(self.limit)[:-1])
        return num * SECONDS_PER_UNIT[unit]


DriftWindowingStrategy = Annotated[
    Union[
        AmountWindowingStrategy,
        TimeWindowingStrategy,
    ],
    Field(discriminator="id"),
]


class DataDriftTriggerConfig(ModynBaseModel):
    id: Literal["DataDriftTrigger"] = Field("DataDriftTrigger")

    detection_interval: Optional[__TriggerConfig] = Field(  # type: ignore[valid-type]
        None, description="The Trigger policy to determine the interval at which drift detection is performed."
    )  # currently not used

    detection_interval_data_points: int = Field(
        1000, description="The number of samples in the interval after which drift detection is performed.", ge=1
    )

    windowing_strategy: DriftWindowingStrategy = Field(
        AmountWindowingStrategy, description="Which windowing strategy to use for current and reference data"
    )

    reset_current_window_on_trigger: bool = Field(
        False, description="Whether the current window should be reset on trigger or rather be extended."
    )

    metrics: dict[str, DriftMetric] = Field(
        min_length=1, description="The metrics used for drift detection keyed by a reference."
    )
    aggregation_strategy: DriftAggregationStrategy = Field(
        MajorityVoteDriftAggregationStrategy(),
        description="The strategy to aggregate the decisions of the individual drift metrics.",
    )
