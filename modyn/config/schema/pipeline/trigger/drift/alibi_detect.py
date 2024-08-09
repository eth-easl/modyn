from typing import Annotated, Literal

from pydantic import Field, model_validator

from modyn.config.schema.base_model import ModynBaseModel


class _AlibiDetectBaseDriftMetric(ModynBaseModel):
    p_val: float = Field(0.05, description="The p-value threshold for the drift detection.")
    x_ref_preprocessed: bool = Field(False)


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
    num_permutations: int | None = Field(
        None,
        description=(
            "The number of permutations used in the MMD hypothesis permutation test. 100 is the alibi-detect default. "
            "None disables hypothesis testing"
        ),
    )
    kernel: str = Field(
        "GaussianRBF", description="The kernel used for distance calculation imported from alibi_detect.utils.pytorch"
    )
    configure_kernel_from_x_ref: bool = Field(True)
    threshold: float | None = Field(
        None,
        description=(
            "When given, we compare the raw distance metric to this threshold to make a triggering decision instead of "
            "using the output of the hypothesis test."
        ),
    )

    @model_validator(mode="after")
    def validate_threshold_permutations(self) -> "AlibiDetectMmdDriftMetric":
        if self.threshold is not None and self.num_permutations is not None:
            raise ValueError(
                "threshold and num_permutations are mutually exclusive."
                + "Please specify whether you want to use hypothesis testing "
                + "or threshold comparison for making drift decisions."
            )

        if self.threshold is None and self.num_permutations is None:
            raise ValueError("Please specify either threshold or num_permutations")

        return self


class AlibiDetectKSDriftMetric(_AlibiDetectBaseDriftMetric, _AlibiDetectAlternativeMixin, _AlibiDetectCorrectionMixin):
    id: Literal["AlibiDetectKSDriftMetric"] = Field("AlibiDetectKSDriftMetric")


class AlibiDetectCVMDriftMetric(_AlibiDetectBaseDriftMetric, _AlibiDetectCorrectionMixin):
    id: Literal["AlibiDetectCVMDriftMetric"] = Field("AlibiDetectCVMDriftMetric")


AlibiDetectDriftMetric = Annotated[
    AlibiDetectMmdDriftMetric | AlibiDetectKSDriftMetric | AlibiDetectCVMDriftMetric,
    Field(discriminator="id"),
]
