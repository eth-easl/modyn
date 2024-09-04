from unittest.mock import MagicMock, patch

import pytest

from modyn.config.schema.pipeline.trigger.simple import SimpleTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.data_amount import (
    DataAmountTriggerConfig,
)
from modyn.supervisor.internal.triggers.amounttrigger import DataAmountTrigger
from modyn.supervisor.internal.triggers.utils.warmuptrigger import WarmupTrigger


@pytest.fixture
def simple_trigger_config() -> SimpleTriggerConfig:
    return DataAmountTriggerConfig(num_samples=3)


def test_warmup_trigger_initialization_without_policy(
    simple_trigger_config: SimpleTriggerConfig,
) -> None:
    warmup_trigger = WarmupTrigger(warmup_intervals=5)

    assert warmup_trigger._delegation_counter == 0
    assert warmup_trigger._warmup_intervals == 0
    assert warmup_trigger.trigger is None
    assert warmup_trigger.completed


def test_warmup_trigger_initialization_with_policy(
    simple_trigger_config: SimpleTriggerConfig,
) -> None:
    warmup_trigger = WarmupTrigger(warmup_intervals=5, warmup_policy=simple_trigger_config)

    assert warmup_trigger._delegation_counter == 0
    assert warmup_trigger._warmup_intervals == 5
    assert isinstance(warmup_trigger.trigger, DataAmountTrigger)
    assert not warmup_trigger.completed


@patch.object(DataAmountTrigger, "inform", return_value=[1])
def test_delegate_inform_without_warmup_trigger(
    mock_trigger_inform: MagicMock,
) -> None:
    warmup_trigger = WarmupTrigger(warmup_intervals=2)
    batch = [(1, 100), (2, 200)]

    result = warmup_trigger.delegate_inform(batch)

    assert warmup_trigger._delegation_counter == 0
    assert result is True
    assert mock_trigger_inform.call_count == 0


@patch.object(DataAmountTrigger, "inform", return_value=[1])
def test_delegate_inform_with_warmup_trigger(
    mock_trigger_inform: MagicMock,
    simple_trigger_config: SimpleTriggerConfig,
) -> None:
    warmup_trigger = WarmupTrigger(warmup_intervals=2, warmup_policy=simple_trigger_config)
    batches = [[(2 * i + 1, 0), (2 * i + 2, 0)] for i in range(3)]
    assert not warmup_trigger.completed

    result = warmup_trigger.delegate_inform(batches[0])
    assert warmup_trigger._delegation_counter == 1
    assert result is True
    mock_trigger_inform.assert_called_once_with([(1, 0, 0), (2, 0, 0)])
    assert not warmup_trigger.completed

    mock_trigger_inform.reset_mock()

    result = warmup_trigger.delegate_inform(batches[1])
    assert warmup_trigger._delegation_counter == 2
    assert result is True
    mock_trigger_inform.assert_called_once_with([(3, 0, 0), (4, 0, 0)])
    assert warmup_trigger.completed

    with pytest.raises(AssertionError):
        warmup_trigger.delegate_inform(batches[2])
