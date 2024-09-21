from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from modyn.config.schema.base_model import ModynBaseModel
from modyn.config.schema.pipeline.evaluation.config import EvalDataConfig
from modyn.config.schema.pipeline.trigger.common.batched import BatchedTriggerConfig
from modyn.config.schema.pipeline.trigger.performance.criterion import (
    PerformanceTriggerCriterion,
    StaticNumberAvoidableMisclassificationCriterion,
)
from modyn.const.types import ForecastingMethod, TriggerEvaluationMode


class PerformanceTriggerEvaluationConfig(ModynBaseModel):
    # TODO(@robinholzi): Support sampling

    device: str = Field(description="The device the model should be put on.")
    dataset: EvalDataConfig = Field(description="The dataset on which the model is evaluated.")
    label_transformer_function: str | None = Field(
        None, description="Function used to transform the label (tensors of integers)."
    )

    @field_validator("dataset")
    @classmethod
    def validate_metrics(cls, dataset: EvalDataConfig) -> EvalDataConfig:
        """Assert that we have at least the accuracy metric."""
        if not any(metric.name == "Accuracy" for metric in dataset.metrics):
            raise ValueError("The accuracy metric is required for the performance trigger.")
        return dataset


class _InternalPerformanceTriggerConfig(BatchedTriggerConfig):
    data_density_window_size: int = Field(
        20,
        description="The window size for the data density estimation. Only used for lookahead mode.",
    )
    performance_triggers_window_size: int = Field(
        10,
        description="The maximum number of evaluations after triggers to consider for computing the expect performance.",
    )

    evaluation: PerformanceTriggerEvaluationConfig = Field(
        description="Configuration for the evaluation of the performance trigger."
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

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, mode: TriggerEvaluationMode) -> TriggerEvaluationMode:
        """Assert that the forecasting method is set if lookahead mode is
        used."""
        if mode == "lookahead":
            raise ValueError("Currently only hindsight mode is supported.")
        return mode


class PerformanceTriggerConfig(_InternalPerformanceTriggerConfig):
    id: Literal["PerformanceTrigger"] = Field("PerformanceTrigger")

    decision_criteria: dict[str, PerformanceTriggerCriterion] = Field(
        description=(
            "The decision criteria to be used for the performance trigger. If any of the criteria is met, "
            "the trigger will be executed. The criteria will be evaluated in the order they are defined. "
            "Every criterion is linked to a metric. Some of the criteria implicitly only work on accuracy which is "
            "the default metric that is always generated and cannot be disabled. To define a "
            "`StaticPerformanceThresholdCriterion` on Accuracy, the evaluation config has to define the accuracy metric."
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_decision_criteria(self) -> "PerformanceTriggerConfig":
        """Assert that all criteria use metrics that are defined in the
        evaluation config."""
        metrics = {metric.name for metric in self.evaluation.dataset.metrics}
        for criterion in self.decision_criteria.values():
            if isinstance(criterion, StaticNumberAvoidableMisclassificationCriterion):
                continue
            if criterion.metric not in metrics:
                raise ValueError(
                    f"Criterion {criterion.id} uses metric {criterion.metric} which is not defined in the evaluation config."
                )
        return self

    @model_validator(mode="after")
    def warmup_policy_requirement(self) -> Self:
        """Assert whether the warmup policy is set when a metric needs
        calibration."""
        for criterion in self.decision_criteria.values():
            if criterion.needs_calibration and self.warmup_policy is None:
                raise ValueError("A warmup policy is required for performance criteria that need calibration.")
        return self
