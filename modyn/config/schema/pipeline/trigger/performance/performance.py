from typing import Literal

from pydantic import Field, field_validator

from modyn.config.schema.base_model import ModynBaseModel
from modyn.config.schema.pipeline.evaluation.config import EvalDataConfig
from modyn.config.schema.pipeline.trigger.performance.criterion import (
    PerformanceTriggerCriterion,
)
from modyn.const.types import ForecastingMethod, TriggerEvaluationMode


class PerformanceTriggerEvaluationConfig(ModynBaseModel):
    max_samples_per_evaluation: int = Field(
        1000,
        description=(
            "The maximum number of samples that should be used for the evaluation. If more are available, "
            "downsampling will be used"
        ),
    )  # TODO: currently not yet supported by the evaluator, we use a evaluation dataset that is sufficiently small

    device: str = Field(description="The device the model should be put on.")
    dataset: EvalDataConfig = Field(description="The dataset on which the model is evaluated.")
    label_transformer_function: str = Field(
        "", description="Function used to transform the label (tensors of integers)."
    )

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, value: EvalDataConfig) -> EvalDataConfig:
        if len(value.metrics) != 1:
            raise ValueError("Only one metric is allowed for performance trigger evaluation.")
        return value


class PerformanceTriggerConfig(ModynBaseModel):
    id: Literal["PerformanceTrigger"] = Field("PerformanceTrigger")
    detection_interval_data_points: int = Field(
        description=(
            "Specifies after how many samples another believe update (query density "
            "estimation, accuracy evaluation) should be performed."
        )
    )

    evaluation: PerformanceTriggerEvaluationConfig = Field(
        description="Configuration for the evaluation of the performance trigger."
    )

    decision_criteria: list[PerformanceTriggerCriterion] = Field(
        description=(
            "The decision criteria to be used for the performance trigger. If any of the criteria is met, "
            "the trigger will be executed."  # TODO: maybe support custom aggregation functions
        ),
        min_length=1,
    )

    mode: TriggerEvaluationMode = Field(
        "hindsight",
        description="Whether to also consider forecasted future performance in the drift decision.",
    )
    forecasting_method: ForecastingMethod = Field(
        "ridge_regression",
        description=(
            "The method to generate the forecasted performance and data density estimates. "
            "Only used for lookahead mode."
        ),
    )

    data_density_window_size: int = Field(
        0,
        description="The window size for the data density estimation. Only used for lookahead mode.",
    )
    performance_triggers_window_size: int = Field(
        10,
        description="The maximum number of evaluations after triggers to consider for computing the expect performance.",
    )
