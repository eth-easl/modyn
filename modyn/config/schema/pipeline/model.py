from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel


class ModelConfig(ModynBaseModel):
    id: str = Field(description="The ID of the model that should be trained.")
    config: dict = Field(
        default_factory=dict,
        description="Configuration dictionary that will be passed to the model on initialization.",
    )
