from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class MatrixEvalTriggerConfig(ModynBaseModel):
    """Base class for EvalTriggerConfigs that allow to evaluate every model to be evaluated at the trigger times"""

    matrix: bool = Field(
        False, description="Weather to evaluate all models at this these points or always only the most recent one"
    )
