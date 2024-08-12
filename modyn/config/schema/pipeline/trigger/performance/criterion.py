from typing import Annotated, Literal, Union

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel

# TODO: accuracy drift detection (like on distances values for drifttrigger)
# TODO: min_metric_value in a separate criterion class


# -------------------------------------------------------------------------------------------------------------------- #
#                                               ExpectedPerformanceConfig                                              #
# -------------------------------------------------------------------------------------------------------------------- #


class StaticExpectedPerformanceConfig(ModynBaseModel):
    id: Literal["StaticExpectedPerformance"] = Field("StaticExpectedPerformance")
    # TODO: assumption: higher is better
    metric_value: float = Field(
        description=(
            "The expected target metric value that the model should achieve. If the performance isn't reached, "
            "more triggers will be executed."
        )
    )


class DynamicExpectedPerformanceConfig(ModynBaseModel):
    id: Literal["DynamicExpectedPerformance"] = Field("DynamicExpectedPerformance")
    num_warmup_evaluation: int = Field(
        description=(
            "For how many warmup triggers should we use a `StaticExpectedPerformanceConfig` in order to "
            "calibrate the expected performance."
        )
    )
    warmup: StaticExpectedPerformanceConfig = Field(description="The warmup configuration.")

    # after warmup:
    trigger_evaluation_averaging_window_size: int = Field(
        3,
        description=(
            "How many of the historic evaluations directly after triggers should we average "
            "to get the expected performance."
        ),
    )


ExpectedPerformanceConfig = Annotated[
    Union[StaticExpectedPerformanceConfig, DynamicExpectedPerformanceConfig],
    Field(discriminator="id"),
]


class _ExpectedPerformanceMixin(ModynBaseModel):
    expected_performance: ExpectedPerformanceConfig = Field(
        description=(
            "The expected performance of the model used to calculate the avoidable misclassifications which "
            "make up the regret of not triggering."
        )
    )


# -------------------------------------------------------------------------------------------------------------------- #
#                                                 PerformanceCriterion                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
# uses the evaluation results directory to derive a triggering decision


class StaticPerformanceThresholdCriterion(ModynBaseModel):
    id: Literal["StaticPerformanceThresholdCriterion"] = Field("StaticPerformanceThresholdCriterion")
    metric_threshold: float = Field(
        0.0,
        description=(
            "The minimum target metric value that the model should achieve. If the performance is NOT reached "
            "a trigger will be forced."
        ),
    )


# TODO: warmup needed
class DynamicPerformanceThresholdCriterion(_ExpectedPerformanceMixin):
    """
    Triggers after comparison of current performance with the a rolling average of historic performances after triggers.
    """

    id: Literal["DynamicPerformanceThresholdCriterion"] = Field("DynamicPerformanceThresholdCriterion")
    rolling_average_window_size: int = Field(
        3,
        description="How many of the historic evaluations directly after triggers should we average.",
    )
    allowed_deviation: float = Field(
        0.05,
        description=(
            "The allowed deviation from the expected performance. Will only trigger if the performance is "
            "below the expected performance minus the allowed deviation."
        ),
    )


# TODO: drift: rolling average + x % (instead of percentile)


# -------------------------------------------------------------------------------------------------------------------- #
#                                           NumberMisclassificationCriterion                                           #
# -------------------------------------------------------------------------------------------------------------------- #


class _NumberAvoidableMisclassificationCriterion(_ExpectedPerformanceMixin):
    """
    Trigger based on the cumulated number of avoidable misclassifications.

    An avoidable misclassification is a misclassification that would have been avoided if a trigger would have been.
    E.g. if we currency see an accuracy of 80% but expect 95% with a trigger, 15% of the samples are avoidable
    misclassifications.

    We estimate the number of avoidable misclassifications with the measured and expected performance.

    The cumulated number of misclassifications can be seen a regret of not triggering.

    Advantage: cumulative metric allows to account for persisting slightly degraded performance. Eventually
    the cumulated number of misclassifications will trigger a trigger.
    """


class StaticNumberAvoidableMisclassificationCriterion(_NumberAvoidableMisclassificationCriterion):
    id: Literal["StaticNumberMisclassificationCriterion"] = Field("StaticNumberMisclassificationCriterion")
    avoidable_misclassification_threshold: float = Field(
        description="The threshold for the misclassification rate that will invoke a trigger."
    )


# class InferredNumberAvoidableMisclassificationCriterion(_NumberAvoidableMisclassificationCriterion):
#     """
#     `misclassification_threshold` will be inferred by the given inputs according to the following scheme:
#     - `dataset_num_samples`: The number of samples in the dataset.
#     - `dataset_time_range`: The time range of the dataset.
#     - `target_num_triggers`: The number of triggers that are expected over the whole dataset assuming
#         constant gradual drift.
#     --> through the `expected_performance` we can infer the expected misclassification rate and thus the threshold
#     """

#     id: Literal["InferredNumberMisclassificationCriterion"] = Field(
#         "InferredNumberMisclassificationCriterion"
#     )

#     dataset_num_samples: int = Field(
#         description="The number of samples in the dataset."
#     )
#     dataset_time_range: int = Field(description="The time range of the dataset in epochs.")
#     target_num_triggers: int = Field(
#         description="The number of triggers that are expected over the whole dataset assuming gradual drift."
#     )
#     expected_performance: float = Field
#     relative_avoidable_misclassification_threshold: float = Field(
#         description=(
#             "The threshold for the triggering misclassification"
#             "The relative misclassification rate is the misclassification rate that would have been avoided "
#             "if a trigger would have been invoked."
#         )
#     )

#     @property
#     def misclassification_threshold(self) -> float:
#         epochs_per_trigger = self.dataset_time_range / self.target_num_triggers

#         return self..metric_value


# -------------------------------------------------------------------------------------------------------------------- #
#                                                         Union                                                        #
# -------------------------------------------------------------------------------------------------------------------- #

PerformanceTriggerCriterion = Annotated[
    Union[
        StaticPerformanceThresholdCriterion,
        DynamicPerformanceThresholdCriterion,
        StaticNumberAvoidableMisclassificationCriterion,
    ],
    Field(discriminator="id"),
]
