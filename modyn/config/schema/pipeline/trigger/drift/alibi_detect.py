from typing import Annotated, Literal, Union

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class _AlibiDetectBaseDriftMetric(ModynBaseModel):
    p_val: float = Field(0.05, description="The p-value threshold for the drift detection.")
    kernel: str  # TODO;
    # x_ref_preprocessed
    # configure_kernel_from_x_ref
    device: str | None = Field(
        None,
        description="The device used for the drift detection.",
        pattern="^(cpu|gpu|cuda:?[0-9]*)$",
    )


class AlibiDetectMmdDriftMetric(_AlibiDetectBaseDriftMetric):
    id: Literal["AlibiDetectMmdDriftMetric"] = Field("AlibiDetectMmdDriftMetric")
    n_permutations: bool = Field(
        100,
        description=("The number of permutations used in the MMD calculation. " "100 is the alibi-detect default."),
    )


class AlibiDetectTODODriftMetric(_AlibiDetectBaseDriftMetric):
    id: Literal["AlibiDetectTODODriftMetric"] = Field("AlibiDetectTODODriftMetric")


AlibiDetectDriftMetric = Annotated[
    Union[AlibiDetectMmdDriftMetric, AlibiDetectTODODriftMetric],
    Field(discriminator="id"),
]
