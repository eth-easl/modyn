from typing import Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel


class DataAmountTriggerConfig(ModynBaseModel):
    id: Literal["DataAmountTrigger"] = Field("DataAmountTrigger")
    num_samples: int = Field(description="The number of samples that should trigger the pipeline.", ge=1)
