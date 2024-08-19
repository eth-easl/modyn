from unittest.mock import MagicMock, patch

import pytest

from modyn.config.schema.pipeline.trigger.ensemble import (
    AtLeastNEnsembleStrategy,
    EnsembleTriggerConfig,
)
from modyn.config.schema.pipeline.trigger.simple.data_amount import (
    DataAmountTriggerConfig,
)
from modyn.config.schema.pipeline.trigger.simple.time import TimeTriggerConfig
from modyn.supervisor.internal.triggers.amounttrigger import DataAmountTrigger
from modyn.supervisor.internal.triggers.ensembletrigger import EnsembleTrigger
from modyn.supervisor.internal.triggers.timetrigger import TimeTrigger
from modyn.supervisor.internal.triggers.trigger import TriggerContext
from modyn.supervisor.internal.triggers.utils.models import TriggerPolicyEvaluationLog


@pytest.fixture
def ensemble_trigger_config() -> EnsembleTriggerConfig:
    return EnsembleTriggerConfig(
        detection_interval_data_points=10,
        subtriggers={
            "time_trigger": TimeTriggerConfig(every="5s"),
            "data_amount_trigger": DataAmountTriggerConfig(num_samples=8),
        },
        ensemble_strategy=AtLeastNEnsembleStrategy(n=1),
    )


@patch("modyn.supervisor.internal.triggers.timetrigger.TimeTrigger", autospec=True)
@patch("modyn.supervisor.internal.triggers.amounttrigger.DataAmountTrigger", autospec=True)
def test_initialization(
    mock_time_trigger: MagicMock,
    mock_data_amount_trigger: MagicMock,
    ensemble_trigger_config: EnsembleTriggerConfig,
) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)
    assert trigger.config.detection_interval_data_points == 10
    assert len(trigger.subtriggers) == 2
    assert isinstance(trigger.subtriggers["time_trigger"], TimeTrigger)
    assert isinstance(trigger.subtriggers["data_amount_trigger"], DataAmountTrigger)


@patch.object(DataAmountTrigger, "init_trigger")
@patch.object(TimeTrigger, "init_trigger")
def test_init_trigger(
    mock_time_trigger_init: MagicMock,
    mock_data_amount_init: MagicMock,
    ensemble_trigger_config: EnsembleTriggerConfig,
) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)
    mock_context = MagicMock(spec=TriggerContext)
    trigger.init_trigger(mock_context)

    # Ensure subtriggers are initialized with the context
    mock_time_trigger_init.assert_called_once_with(mock_context)
    mock_data_amount_init.assert_called_once_with(mock_context)


@patch.object(DataAmountTrigger, "inform", side_effect=[iter([1]), iter([])])
@patch.object(TimeTrigger, "inform", side_effect=[iter([2]), iter([])])
def test_inform_trigger(
    mock_time_trigger_init: MagicMock,
    mock_data_amount_init: MagicMock,
    ensemble_trigger_config: EnsembleTriggerConfig,
) -> None:
    ensemble_trigger_config.detection_interval_data_points = 3
    trigger = EnsembleTrigger(ensemble_trigger_config)

    # Run the inform method: 2 batches --> 2 calls to each subtrigger's inform method
    trigger_log = TriggerPolicyEvaluationLog()
    results = list(trigger.inform([(i, i + 100, 1) for i in range(6)], log=trigger_log))

    assert len(results) == 1
    assert results[0] == 2
    assert len(trigger_log.evaluations) == 2
    assert trigger_log.evaluations[0].triggered
    assert not trigger_log.evaluations[1].triggered


@patch.object(DataAmountTrigger, "inform_previous_model")
@patch.object(TimeTrigger, "inform_previous_model")
def test_inform_previous_model(
    mock_time_trigger_inform_previous_model: MagicMock,
    mock_data_amount_inform_previous_model: MagicMock,
    ensemble_trigger_config: EnsembleTriggerConfig,
) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)
    mock_model_id = 99
    trigger.inform_previous_model(mock_model_id)
    mock_time_trigger_inform_previous_model.assert_called_once_with(mock_model_id)
    mock_data_amount_inform_previous_model.assert_called_once_with(mock_model_id)
