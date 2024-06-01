from modyn.config.schema.pipeline.pipeline import DataAmountTriggerConfig
from modyn.supervisor.internal.triggers import DataAmountTrigger


def test_initialization() -> None:
    trigger = DataAmountTrigger(DataAmountTriggerConfig(num_samples=42))
    assert trigger.data_points_for_trigger == 42
    assert trigger.remaining_data_points == 0


def test_inform() -> None:
    trigger = DataAmountTrigger(DataAmountTriggerConfig(num_samples=1))
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert list(trigger.inform([])) == []
    assert list(trigger.inform([(10, 1)])) == [0]
    assert list(trigger.inform([(10, 1), (10, 1)])) == [0, 1]
    assert list(trigger.inform([(10, 1), (10, 1), (10, 1)])) == [0, 1, 2]

    trigger = DataAmountTrigger(DataAmountTriggerConfig(num_samples=2))
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert list(trigger.inform([(10, 1)])) == []
    assert list(trigger.inform([(10, 1)])) == [0]
    assert list(trigger.inform([(10, 1), (10, 1)])) == [1]
    assert list(trigger.inform([(10, 1), (10, 1), (10, 1), (10, 1)])) == [1, 3]
    assert list(trigger.inform([(10, 1), (10, 1), (10, 1)])) == [1]
    assert list(trigger.inform([(10, 1)])) == [0]
    assert list(trigger.inform([(10, 1), (10, 1), (10, 1), (10, 1), (10, 1)])) == [1, 3]
    assert list(trigger.inform([(10, 1)])) == [0]

    trigger = DataAmountTrigger(DataAmountTriggerConfig(num_samples=5))
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert list(trigger.inform([(10, 1), (10, 1), (10, 1), (10, 1)])) == []
    assert list(trigger.inform([(10, 1), (10, 1), (10, 1)])) == [0]
    assert list(trigger.inform([(10, 1), (10, 1), (10, 1)])) == [2]
