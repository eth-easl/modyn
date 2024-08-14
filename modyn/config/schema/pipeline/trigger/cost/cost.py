"""Cost aware triggers are an extension of performance aware triggers.

They not only evaluate data density and model performance but also
consider the cost of a trigger both in terms of wall clock time and
number of triggers.
"""

from typing import Literal

from pydantic import Field

from modyn.config.schema.pipeline.trigger.performance.performance import (
    _InternalPerformanceTriggerConfig,
)


class CostTriggerConfig(_InternalPerformanceTriggerConfig):
    """Offers the same set of `decision_criteria` as `PerformanceTriggerConfig`
    but implicitly adds a cost criterion to the list.

    We use the `_InternalPerformanceTriggerConfig` base class as we cannot override the `id` field of
    `PerformanceTriggerConfig`.

    The cost criterion is met if:
    - `mode` is `hindsight` and the expected cost of the trigger is below the cumulated regret which
        resulted from avoidable misclassifications through not triggering
    """

    id: Literal["CostTrigger"] = Field("CostTrigger")

    misclassification_per_training_time: float = Field(
        description="How many misclassifications are we willing to accept per second of training time saved."
    )
