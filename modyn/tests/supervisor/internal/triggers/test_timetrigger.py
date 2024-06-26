from modyn.config.schema.pipeline import TimeTriggerConfig
from modyn.supervisor.internal.triggers import TimeTrigger


def test_initialization() -> None:
    trigger = TimeTrigger(TimeTriggerConfig(every="2s"))
    assert trigger.config.every_seconds == 2
    assert trigger.next_trigger_at is None


def test_inform() -> None:
    trigger = TimeTrigger(TimeTriggerConfig(every="1000s"))
    LABEL = 2  # pylint: disable=invalid-name
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert list(trigger.inform([])) == []
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert list(trigger.inform([(10, 0, LABEL)])) == []
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert list(trigger.inform([(10, 500, LABEL)])) == []
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert list(trigger.inform([(10, 999, LABEL)])) == []
    assert list(trigger.inform([(10, 1000, LABEL)])) == [-1]  # Trigger includes 0, 500, 900, but not 1000
    assert list(trigger.inform([(10, 1500, LABEL), (10, 1600, LABEL), (10, 2000, LABEL)])) == [
        1
    ]  # 2000 enables us to know that 1600 should trigger!
    assert list(trigger.inform([(10, 3000, LABEL), (10, 4000, LABEL)])) == [-1, 0]
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert list(trigger.inform([(10, 4100, LABEL), (10, 4200, LABEL)])) == []
    assert list(trigger.inform([(10, 5000, LABEL)])) == [-1]
    assert list(trigger.inform([(10, 6000, LABEL), (10, 7000, LABEL), (10, 8000, LABEL), (10, 9000, LABEL)])) == [
        -1,
        0,
        1,
        2,
    ]
    assert list(trigger.inform([(10, 15000, LABEL)])) == [-1, -1, -1, -1, -1, -1]
    assert list(trigger.inform([(10, 17000, LABEL), (10, 18000, LABEL)])) == [-1, -1, 0]
