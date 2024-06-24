from typing import Any, Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel

from .config import TriggerConfig


class DataDriftTriggerConfig(ModynBaseModel):
    id: Literal["DataDriftTrigger"] = Field("DataDriftTrigger")
    detection_interval: TriggerConfig
    detection_interval_data_points: int = Field(
        1000, description="The number of samples in the interval after which drift detection is performed.", ge=1
    )
    sample_size: int | None = Field(None, description="The number of samples used for the metric calculation.", ge=1)
    metric: str = Field("model", description="The metric used for drift detection.")
    metric_config: dict[str, Any] = Field(default_factory=dict, description="Configuration for the evidently metric.")
