from typing import Literal

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class AfterTrainingEvalTriggerConfig(ModynBaseModel):
    mode: Literal["after_training"] = Field("after_training")
