from __future__ import annotations

from typing import Annotated, ForwardRef, Literal, Optional, Union

from modyn.config.schema.base_model import ModynBaseModel
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


class DataDriftTriggerConfig(ModynBaseModel):
    id: Literal["DataDriftTrigger"] = Field("DataDriftTrigger")

    detection_interval: Optional[__TriggerConfig] = Field(  # type: ignore[valid-type]
        None, description="The Trigger policy to determine the interval at which drift detection is performed."
    )  # currently not used

    detection_interval_data_points: int = Field(
        1000, description="The number of samples in the interval after which drift detection is performed.", ge=1
    )
    sample_size: int | None = Field(None, description="The number of samples used for the metric calculation.", ge=1)

    metrics: dict[str, DriftMetric] = Field(
        min_length=1, description="The metrics used for drift detection keyed by a reference."
    )
    aggregation_strategy: DriftAggregationStrategy = Field(
        MajorityVoteDriftAggregationStrategy(),
        description="The strategy to aggregate the decisions of the individual drift metrics.",
    )
