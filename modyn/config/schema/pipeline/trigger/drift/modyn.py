from typing import Annotated, Literal, Union

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class ModynMmdDriftMetric(ModynBaseModel):
    id: Literal["ModynMmdDriftMetric"] = Field("ModynMmdDriftMetric")
    bootstrap: bool = Field(False)
    quantile_probability: float = 0.05
    num_bootstraps: int = Field(100)
    num_pca_component: int | None = Field(None)
    device: str = Field("cpu", description="Pytorch device to use for computation")
    num_workers: int = Field(1, description="Number of workers to use for computation")


ModynDriftMetric = Annotated[
    Union[ModynMmdDriftMetric],
    Field(discriminator="id"),
]
