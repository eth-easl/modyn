from modyn.config.schema.base_model import ModynBaseModel

from .criterion import DriftDecisionCriterion


class BaseMetric(ModynBaseModel):
    decision_criterion: DriftDecisionCriterion
