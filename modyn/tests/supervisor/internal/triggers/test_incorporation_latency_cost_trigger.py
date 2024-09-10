from unittest.mock import MagicMock, call, patch

from pytest import fixture

from modyn.config.schema.pipeline.trigger.cost.cost import (
    DataIncorporationLatencyCostTriggerConfig,
)
from modyn.config.schema.pipeline.trigger.simple.data_amount import (
    DataAmountTriggerConfig,
)
from modyn.supervisor.internal.triggers.cost.cost_tracker import CostTracker
from modyn.supervisor.internal.triggers.dataincorporationlatency_costtrigger import (
    DataIncorporationLatencyCostTrigger,
)
from modyn.supervisor.internal.triggers.trigger import TriggerContext


@fixture
def latency_trigger_config() -> DataIncorporationLatencyCostTriggerConfig:
    return DataIncorporationLatencyCostTriggerConfig(
        evaluation_interval_data_points=4,
        cost_tracking_window_size=5,
        incorporation_delay_per_training_second=1.0,
    )


def test_initialization(
    latency_trigger_config: DataIncorporationLatencyCostTriggerConfig,
    dummy_trigger_context: TriggerContext,
) -> None:
    trigger = DataIncorporationLatencyCostTrigger(latency_trigger_config)
    assert trigger._sample_left_until_detection == latency_trigger_config.evaluation_interval_data_points
    assert not trigger._triggered_once
    assert trigger._previous_batch_end_time is None
    assert trigger._unincorporated_samples == 0

    assert trigger.cost_tracker.measurements.maxlen == latency_trigger_config.cost_tracking_window_size
    assert trigger.latency_tracker.cumulative_latency_regret == 0.0

    assert trigger.context is None
    trigger.init_trigger(dummy_trigger_context)
    assert trigger.context == dummy_trigger_context


@patch.object(CostTracker, "inform_trigger")
@patch.object(CostTracker, "forecast_training_time", side_effect=[10.0, 10.0])
@patch.object(
    DataIncorporationLatencyCostTrigger,
    "_compute_regret_metric",
    side_effect=[(regret, {}) for regret in [-1, 4.0, 11.0]],
)
def test_inform_and_new_model(
    mock_compute_regret: MagicMock,
    mock_forecast_training_time: MagicMock,
    mock_inform_trigger: MagicMock,
    latency_trigger_config: DataIncorporationLatencyCostTriggerConfig,
) -> None:
    trigger = DataIncorporationLatencyCostTrigger(latency_trigger_config)
    assert not trigger._triggered_once

    # detection interval 1: at sample (3, 103) --> forced trigger
    trigger_results = list(trigger.inform([(i, 100 + i, 0) for i in range(6)]))
    assert len(trigger_results) == 1
    assert trigger_results == [3]
    assert trigger._triggered_once
    assert mock_compute_regret.call_count == 1
    assert mock_forecast_training_time.call_count == 0  # first trigger is forced, not training time forecast possible
    assert trigger._leftover_data == [(i, 100 + i) for i in range(4, 6)]

    trigger.inform_new_model(1, number_samples=4, training_time=10.0)
    assert trigger._unincorporated_samples == 0
    assert mock_inform_trigger.call_count == 1
    assert mock_inform_trigger.call_args_list == [call(4, 10.0)]

    # reset mocks
    mock_compute_regret.reset_mock()
    mock_inform_trigger.reset_mock()

    # inform about 5 more samples
    # detection interval 2: at sample (7, 107): 4 samples --> expect 10sek training; cumulative regret=4
    # 4 * 1s < 10s --> no trigger, 3 samples leftover
    trigger_results = list(trigger.inform([(i, 100 + i, 0) for i in range(6, 11)]))
    assert len(trigger_results) == 0
    assert trigger._triggered_once
    assert mock_compute_regret.call_count == 1
    assert mock_forecast_training_time.call_count == 1
    assert trigger._leftover_data == [(i, 100 + i) for i in range(8, 11)]
    assert trigger._unincorporated_samples == 4

    assert mock_inform_trigger.call_count == 0

    # reset mocks
    mock_compute_regret.reset_mock()
    mock_forecast_training_time.reset_mock()

    # inform about 1 more sample
    # detection interval 3: at sample (11, 111) --> expect 10sek training; cumulative regret=11
    # no trigger (because of decision policy), 0 samples leftover
    trigger_results = list(trigger.inform([(i, 100 + i, 0) for i in range(11, 12)]))
    assert len(trigger_results) == 1
    assert trigger_results == [0]
    assert len(trigger._leftover_data) == 0
    assert trigger._sample_left_until_detection == 4
    assert mock_compute_regret.call_count == 1
    assert mock_forecast_training_time.call_count == 1


@patch.object(CostTracker, "forecast_training_time", return_value=0)
@patch.object(
    DataIncorporationLatencyCostTrigger,
    "_compute_regret_metric",
    return_value=(10000, {}),
)
def test_warmup_trigger(
    mock_compute_regret: MagicMock,
    mock_forecast_training_time: MagicMock,
    latency_trigger_config: DataIncorporationLatencyCostTriggerConfig,
) -> None:
    latency_trigger_config.evaluation_interval_data_points = 1
    latency_trigger_config.warmup_intervals = 3
    latency_trigger_config.warmup_policy = DataAmountTriggerConfig(num_samples=2)
    trigger = DataIncorporationLatencyCostTrigger(latency_trigger_config)
    assert not trigger._triggered_once

    trigger_results = list(trigger.inform([(0, 0, 0)]))
    assert trigger_results == [0]  # first trigger is enforced
    assert trigger._triggered_once
    assert mock_compute_regret.call_count == 1  # not on first trigger, no model
    assert mock_forecast_training_time.call_count == 0

    mock_compute_regret.reset_mock()
    mock_forecast_training_time.reset_mock()

    trigger_results = list(trigger.inform([(1, 0, 0)]))
    assert trigger_results == [0]
    assert trigger._triggered_once
    assert mock_compute_regret.call_count == 1
    assert mock_forecast_training_time.call_count == 0

    mock_compute_regret.reset_mock()
    mock_forecast_training_time.reset_mock()

    trigger_results = list(trigger.inform([(2, 0, 0)]))
    assert len(trigger_results) == 0
    assert trigger._triggered_once
    assert mock_compute_regret.call_count == 1
    assert mock_forecast_training_time.call_count == 0

    mock_compute_regret.reset_mock()
    mock_forecast_training_time.reset_mock()

    trigger_results = list(trigger.inform([(3, 0, 0)]))
    assert trigger_results == [0]
    assert trigger._triggered_once
    assert mock_compute_regret.call_count == 1
    assert mock_forecast_training_time.call_count == 1
