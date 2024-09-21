"""We will mock CostTrigger here as the basic integration between CostTrigger
and it's children is already asserted through
`AvoidableMisclassificationCostTrigger`."""

from unittest.mock import ANY, MagicMock, patch

from pytest import fixture

from modyn.config.schema.pipeline.evaluation.config import EvalDataConfig
from modyn.config.schema.pipeline.evaluation.metrics import (
    AccuracyMetricConfig,
    F1ScoreMetricConfig,
)
from modyn.config.schema.pipeline.trigger.cost.cost import (
    AvoidableMisclassificationCostTriggerConfig,
)
from modyn.config.schema.pipeline.trigger.performance.performance import (
    PerformanceTriggerEvaluationConfig,
)
from modyn.config.schema.pipeline.trigger.simple.data_amount import DataAmountTriggerConfig
from modyn.supervisor.internal.triggers.avoidablemissclassification_costtrigger import (
    AvoidableMisclassificationCostTrigger,
)
from modyn.supervisor.internal.triggers.cost.incorporation_latency_tracker import (
    IncorporationLatencyTracker,
)
from modyn.supervisor.internal.triggers.costtrigger import CostTrigger
from modyn.supervisor.internal.triggers.performance.misclassification_estimator import (
    NumberAvoidableMisclassificationEstimator,
)
from modyn.supervisor.internal.triggers.performance.performancetrigger_mixin import (
    PerformanceTriggerMixin,
)
from modyn.supervisor.internal.triggers.trigger import TriggerContext


@fixture
def trigger_config() -> AvoidableMisclassificationCostTriggerConfig:
    return AvoidableMisclassificationCostTriggerConfig(
        # performance trigger mixin
        evaluation_interval_data_points=42,
        data_density_window_size=100,
        performance_triggers_window_size=10,
        evaluation=PerformanceTriggerEvaluationConfig(
            device="cuda:0",
            dataset=EvalDataConfig(
                dataset_id="dummy_dataset",
                bytes_parser_function="def bytes_parser_function(data: bytes) -> bytes:\n\treturn data",
                batch_size=64,
                dataloader_workers=1,
                metrics=[
                    AccuracyMetricConfig(evaluation_transformer_function=None),
                    F1ScoreMetricConfig(
                        evaluation_transformer_function=None,
                        num_classes=10,
                        average="macro",
                        pos_label=1,
                    ),
                ],
            ),
            label_transformer_function=None,
        ),
        mode="hindsight",
        forecasting_method="rolling_average",
        # cost_trigger
        cost_tracking_window_size=5,
        avoidable_misclassification_latency_per_training_second=1.0,
        warmup_policy=DataAmountTriggerConfig(num_samples=1),
        warmup_intervals=1,
    )


@patch.object(PerformanceTriggerMixin, "_init_trigger")
@patch.object(CostTrigger, "init_trigger")
def test_initialization(
    mock_cost_trigger_init: MagicMock,
    mock_performance_trigger_mixin_init: MagicMock,
    trigger_config: AvoidableMisclassificationCostTriggerConfig,
    dummy_trigger_context: TriggerContext,
) -> None:
    trigger = AvoidableMisclassificationCostTrigger(trigger_config)
    assert trigger._sample_left_until_detection == trigger_config.evaluation_interval_data_points
    assert not trigger._triggered_once
    assert trigger._previous_batch_end_time is None
    assert trigger._unincorporated_samples == 0

    assert isinstance(trigger.misclassification_estimator, NumberAvoidableMisclassificationEstimator)

    assert trigger.cost_tracker.measurements.maxlen == trigger_config.cost_tracking_window_size
    assert trigger.latency_tracker.cumulative_latency_regret == 0.0

    assert trigger.context is None
    trigger.init_trigger(dummy_trigger_context)
    mock_cost_trigger_init.assert_called_once_with(trigger, dummy_trigger_context)
    mock_performance_trigger_mixin_init.assert_called_once_with(trigger, dummy_trigger_context)


@patch.object(PerformanceTriggerMixin, "_inform_new_model")
@patch.object(CostTrigger, "inform_new_model")
def test_inform_new_model(
    mock_cost_trigger_inform_new_model: MagicMock,
    mock_performance_trigger_mixin_inform_new_model: MagicMock,
    trigger_config: AvoidableMisclassificationCostTriggerConfig,
) -> None:
    trigger = AvoidableMisclassificationCostTrigger(trigger_config)

    trigger.inform_new_model(42, 43, 44.0)
    mock_cost_trigger_inform_new_model.assert_called_once_with(trigger, 42, 43, 44.0)
    mock_performance_trigger_mixin_inform_new_model.assert_called_once_with(trigger, 42, ANY)


@patch.object(IncorporationLatencyTracker, "add_latency", return_value=42.0)
@patch.object(
    NumberAvoidableMisclassificationEstimator,
    "estimate_avoidable_misclassifications",
    return_value=(5, -100),
)
@patch.object(PerformanceTriggerMixin, "_run_evaluation", side_effect=[(1, 5, 2, {"Accuracy": 0.6})])
def test_compute_regret_metric(
    mock_run_evaluation: MagicMock,
    mock_estimate_avoidable_misclassifications: MagicMock,
    mock_add_latency: MagicMock,
    trigger_config: AvoidableMisclassificationCostTriggerConfig,
):
    trigger = AvoidableMisclassificationCostTrigger(trigger_config)

    batch = [(i, 100 + i) for i in range(5)]
    batch_duration = 99
    new_regret_latency, _ = trigger._compute_regret_metric(batch, 0, batch_duration)

    mock_run_evaluation.assert_called_once_with(interval_data=batch)

    mock_estimate_avoidable_misclassifications.assert_called_once()
    assert new_regret_latency == 42.0
    mock_add_latency.assert_called_once_with(5, 99)
