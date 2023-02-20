import pytest
from modyn.backend.supervisor.internal.triggers import TimeTrigger


def test_initialization() -> None:
    trigger = TimeTrigger({"trigger_every": "2s"})
    assert trigger.trigger_every_ms == 2000
    assert trigger.next_trigger_at is None


def test_init_fails_if_invalid() -> None:
    with pytest.raises(ValueError, match="Trigger config is missing `trigger_every` field"):
        TimeTrigger({})

    with pytest.raises(ValueError, match="trigger_every must be > 0, but is 0"):
        TimeTrigger({"trigger_every": "0s"})


def test_inform() -> None:
    trigger = TimeTrigger({"trigger_every": "1s"})
    LABEL = 2
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert trigger.inform([]) == []
    assert trigger.inform([("a", 0, LABEL)]) == []
    assert trigger.inform([("a", 500, LABEL)]) == []
    assert trigger.inform([("a", 999, LABEL)]) == []
    assert trigger.inform([("a", 1000, LABEL)]) == [-1] # Trigger includes 0, 500, 900, but not 1000
    assert trigger.inform([("a", 1500, LABEL), ("a", 1600, LABEL), ("a", 2000, LABEL)]) == [1] # 2000 enables us to know that 1600 should trigger!
    assert trigger.inform([("a", 3000, LABEL), ("a", 4000, LABEL)]) == [-1, 0]
    assert trigger.inform([("a", 4100, LABEL), ("a", 4200, LABEL)]) == []
    assert trigger.inform([("a", 5000, LABEL)]) == [-1]
    assert trigger.inform([("a", 6000, LABEL), ("a", 7000, LABEL), ("a", 8000, LABEL), ("a", 9000, LABEL)]) == [-1, 0, 1, 2]