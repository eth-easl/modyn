from typing import Literal

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel


class BetweenTwoTriggersEvalStrategyConfig(ModynBaseModel):
    """This evaluation strategy will evaluate the model on the unique interval
    between two consecutive triggers.

    This exactly reflects the time span where one model is used for
    inference.
    """

    type: Literal["BetweenTwoTriggersEvalStrategy"] = Field("BetweenTwoTriggersEvalStrategy")
