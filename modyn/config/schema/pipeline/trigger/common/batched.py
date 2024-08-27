from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel


class BatchedTriggerConfig(ModynBaseModel):
    evaluation_interval_data_points: int = Field(
        description=(
            "Specifies after how many samples another believe update (query density "
            "estimation, accuracy evaluation, drift detection, ...) should be performed."
        )
    )
