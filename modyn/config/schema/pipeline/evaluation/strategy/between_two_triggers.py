from typing import Literal

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class BetweenTwoTriggersEvalStrategyConfig(ModynBaseModel):
    """This evaluation strategy will evaluate the model on the intervals between two consecutive triggers.

    This exactly reflects the time span where one model is used for inference.
    """

    type: Literal["BetweenTwoTriggersEvalStrategy"] = Field("BetweenTwoTriggersEvalStrategy")
