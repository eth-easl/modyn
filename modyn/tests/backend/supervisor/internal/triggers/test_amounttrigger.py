import pytest
from modyn.backend.supervisor.internal.triggers import DataAmountTrigger


def test_initialization() -> None:
    trigger = DataAmountTrigger({"data_points_for_trigger": 42})
    assert trigger.data_points_for_trigger == 42
    assert trigger.remaining_data_points == 0


def test_init_fails_if_invalid() -> None:
    with pytest.raises(ValueError, match="Trigger config is missing `data_points_for_trigger` field"):
        DataAmountTrigger({})

    with pytest.raises(AssertionError, match="data_points_for_trigger needs to be at least 1"):
        DataAmountTrigger({"data_points_for_trigger": 0})


def test_inform() -> None:
    trigger = DataAmountTrigger({"data_points_for_trigger": 1})
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert trigger.inform([]) == []
    assert trigger.inform([("a", 1)]) == [0]
    assert trigger.inform([("a", 1), ("a", 1)]) == [0, 1]
    assert trigger.inform([("a", 1), ("a", 1), ("a", 1)]) == [0, 1, 2]

    trigger = DataAmountTrigger({"data_points_for_trigger": 2})
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert trigger.inform([("a", 1)]) == []
    assert trigger.inform([("a", 1)]) == [0]
    assert trigger.inform([("a", 1), ("a", 1)]) == [1]
    assert trigger.inform([("a", 1), ("a", 1), ("a", 1), ("a", 1)]) == [1, 3]
    assert trigger.inform([("a", 1), ("a", 1), ("a", 1)]) == [1]
    assert trigger.inform([("a", 1)]) == [0]
    assert trigger.inform([("a", 1), ("a", 1), ("a", 1), ("a", 1), ("a", 1)]) == [1, 3]
    assert trigger.inform([("a", 1)]) == [0]

    trigger = DataAmountTrigger({"data_points_for_trigger": 5})
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert trigger.inform([("a", 1), ("a", 1), ("a", 1), ("a", 1)]) == []
    assert trigger.inform([("a", 1), ("a", 1), ("a", 1)]) == [0]
    assert trigger.inform([("a", 1), ("a", 1), ("a", 1)]) == [2]
