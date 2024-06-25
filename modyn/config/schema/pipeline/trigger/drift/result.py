from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class MetricResult(ModynBaseModel):
    metric_id: str = Field(description="The id of the metric used for drift detection.")
    is_drift: bool
    p_val: float | None = None
    distance: float
    threshold: float | None = None
