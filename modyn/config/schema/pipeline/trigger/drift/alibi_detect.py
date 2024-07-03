from typing import Annotated, Literal, Union

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class _AlibiDetectBaseDriftMetric(ModynBaseModel):
    p_val: float = Field(0.05, description="The p-value threshold for the drift detection.")

    # to be added in the future:

    # x_ref_preprocessed
    # preprocess_at_init
    # configure_kernel_from_x_ref


class AlibiDetectDeviceMixin(ModynBaseModel):
    device: str | None = Field(
        None,
        description="The device used for the drift detection.",
        pattern="^(cpu|gpu|cuda:?[0-9]*)$",
    )


class _AlibiDetectCorrectionMixin(ModynBaseModel):
    correction: Literal["bonferroni", "fdr"] = Field("bonferroni")


class _AlibiDetectAlternativeMixin(ModynBaseModel):
    alternative_hypothesis: Literal["two-sided", "less", "greater"] = Field("two-sided")


class AlibiDetectMmdDriftMetric(_AlibiDetectBaseDriftMetric, AlibiDetectDeviceMixin):
    id: Literal["AlibiDetectMmdDriftMetric"] = Field("AlibiDetectMmdDriftMetric")
    num_permutations: int = Field(
        100,
        description=(
            "The number of permutations used in the MMD hypothesis permutation test. 100 is the alibi-detect default."
        ),
    )
    kernel: str = Field(
        "GaussianRBF", description="The kernel used for distance calculation imported from alibi_detect.utils.pytorch"
    )


class AlibiDetectKSDriftMetric(_AlibiDetectBaseDriftMetric, _AlibiDetectAlternativeMixin, _AlibiDetectCorrectionMixin):
    id: Literal["AlibiDetectKSDriftMetric"] = Field("AlibiDetectKSDriftMetric")


class AlibiDetectCVMDriftMetric(_AlibiDetectBaseDriftMetric, _AlibiDetectCorrectionMixin):
    id: Literal["AlibiDetectCVMDriftMetric"] = Field("AlibiDetectCVMDriftMetric")


AlibiDetectDriftMetric = Annotated[
    Union[AlibiDetectMmdDriftMetric, AlibiDetectKSDriftMetric, AlibiDetectCVMDriftMetric],
    Field(discriminator="id"),
]
