import pytest
from pydantic import ValidationError

from modyn.config.schema.pipeline.trigger.ensemble import (
    AtLeastNEnsembleStrategy,
    CustomEnsembleStrategy,
    MajorityVoteEnsembleStrategy,
)


def test_majority_vote_ensemble_strategy() -> None:
    strategy = MajorityVoteEnsembleStrategy()
    decisions = {"trigger1": True, "trigger2": False, "trigger3": True}
    assert strategy.aggregate_decision_func(decisions)

    decisions = {"trigger1": True, "trigger2": False, "trigger3": False}
    assert not strategy.aggregate_decision_func(decisions)

    decisions = {"trigger1": True, "trigger2": True, "trigger3": True}
    assert strategy.aggregate_decision_func(decisions)

    decisions = {"trigger1": False, "trigger2": False, "trigger3": False}
    assert not strategy.aggregate_decision_func(decisions)


def test_at_least_n_ensemble_strategy() -> None:
    strategy = AtLeastNEnsembleStrategy(n=2)

    decisions = {"trigger1": True, "trigger2": False, "trigger3": True}
    assert strategy.aggregate_decision_func(decisions)

    decisions = {"trigger1": True, "trigger2": False, "trigger3": False}
    assert not strategy.aggregate_decision_func(decisions)

    decisions = {"trigger1": True, "trigger2": True, "trigger3": False}
    assert strategy.aggregate_decision_func(decisions)

    # Test for invalid 'n' (e.g., less than 1)
    with pytest.raises(ValidationError):
        AtLeastNEnsembleStrategy(n=0)


def test_custom_ensemble_strategy() -> None:
    def custom_func(decisions: dict[str, bool]) -> bool:
        return all(decisions.values())

    strategy = CustomEnsembleStrategy(aggregation_function=custom_func)

    decisions = {"trigger1": True, "trigger2": False, "trigger3": True}
    assert not strategy.aggregate_decision_func(decisions)

    decisions = {"trigger1": True, "trigger2": True, "trigger3": True}
    assert strategy.aggregate_decision_func(decisions)

    # Test invalid initialization (e.g., missing aggregation_function)
    with pytest.raises(ValidationError):
        CustomEnsembleStrategy()
