from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel
from modyn.config.schema.pipeline.trigger.simple import SimpleTriggerConfig


class BatchedTriggerConfig(ModynBaseModel):
    evaluation_interval_data_points: int = Field(
        description=(
            "Specifies after how many samples another believe update (query density "
            "estimation, accuracy evaluation, drift detection, ...) should be performed."
        )
    )
    warmup_intervals: int | None = Field(
        None,
        description=(
            "The number of intervals before starting to use the main policy. Some "
            "Policies use this to calibrate a threshold. During the warmup, a simpler `warmup_policy` "
            "is consulted for the triggering decision."
        ),
    )
    warmup_policy: SimpleTriggerConfig | None = Field(
        None,
        description=(
            "The policy to use for triggering during the warmup phase of the main policy. "
            "Policies that don't need calibration can ignore this."
        ),
    )
