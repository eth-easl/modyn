import pytest
from modyn.backend.supervisor.internal.triggers import DataAmountTrigger


def test_initialization() -> None:
    def callback():
        pass

    trigger = DataAmountTrigger(callback, {"data_points_for_trigger": 42})
    assert trigger.data_points_for_trigger == 42
    assert trigger.seen_data_points == 0
    assert trigger.callback == callback  # pylint: disable=comparison-with-callable


def test_init_fails_if_invalid() -> None:
    def callback():
        pass

    with pytest.raises(AssertionError, match="Trigger config is missing `data_points_for_trigger` field"):
        DataAmountTrigger(callback, {})

    with pytest.raises(AssertionError, match="data_points_for_trigger needs to be at least 1"):
        DataAmountTrigger(callback, {"data_points_for_trigger": 0})


def test__decide_for_trigger() -> None:
    def callback():
        pass

    trigger = DataAmountTrigger(callback, {"data_points_for_trigger": 1})
    assert trigger._decide_for_trigger([]) == 0
    assert trigger._decide_for_trigger([("a", 1)]) == 1
    assert trigger._decide_for_trigger([("a", 1), ("a", 1)]) == 2

    trigger = DataAmountTrigger(callback, {"data_points_for_trigger": 2})
    assert trigger._decide_for_trigger([("a", 1)]) == 0
    assert trigger._decide_for_trigger([("a", 1)]) == 1
    assert trigger._decide_for_trigger([("a", 1), ("a", 1)]) == 1
    assert trigger._decide_for_trigger([("a", 1), ("a", 1), ("a", 1), ("a", 1)]) == 2
    assert trigger._decide_for_trigger([("a", 1), ("a", 1), ("a", 1)]) == 1
    assert trigger._decide_for_trigger([("a", 1)]) == 1
