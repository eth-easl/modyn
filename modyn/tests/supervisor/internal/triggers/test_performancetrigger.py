import os
import pathlib
from unittest.mock import MagicMock, call, patch

from pytest import fixture

from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.config.schema.pipeline.evaluation.config import EvalDataConfig
from modyn.config.schema.pipeline.evaluation.metrics import (
    AccuracyMetricConfig,
    F1ScoreMetricConfig,
)
from modyn.config.schema.pipeline.trigger.performance.criterion import (
    DynamicPerformanceThresholdCriterion,
    StaticNumberAvoidableMisclassificationCriterion,
    StaticPerformanceThresholdCriterion,
)
from modyn.config.schema.pipeline.trigger.performance.performance import (
    PerformanceTriggerConfig,
    PerformanceTriggerEvaluationConfig,
)
from modyn.config.schema.system.config import ModynConfig
from modyn.supervisor.internal.triggers.performance.data_density_tracker import (
    DataDensityTracker,
)
from modyn.supervisor.internal.triggers.performance.decision_policy import (
    DynamicPerformanceThresholdDecisionPolicy,
    PerformanceDecisionPolicy,
    StaticNumberAvoidableMisclassificationDecisionPolicy,
    StaticPerformanceThresholdDecisionPolicy,
)
from modyn.supervisor.internal.triggers.performance.performancetrigger_mixin import (
    PerformanceTriggerMixin,
)
from modyn.supervisor.internal.triggers.performancetrigger import (
    PerformanceTrigger,
    _setup_decision_policies,
)
from modyn.supervisor.internal.triggers.trigger import TriggerContext

PIPELINE_ID = 42
SAMPLE = (10, 1, 1)
BASEDIR = pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir"


@fixture
def performance_trigger_config() -> PerformanceTriggerConfig:
    return PerformanceTriggerConfig(
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
        decision_criteria={
            "static_threshold": StaticPerformanceThresholdCriterion(metric="Accuracy", metric_threshold=0.8),
        },
        mode="hindsight",
        forecasting_method="rolling_average",
    )


@fixture
def static_criterion() -> StaticPerformanceThresholdCriterion:
    return StaticPerformanceThresholdCriterion(metric="Accuracy", metric_threshold=0.8)


@fixture
def dynamic_criterion() -> DynamicPerformanceThresholdCriterion:
    return DynamicPerformanceThresholdCriterion(metric="Accuracy", allowed_deviation=0.1)


@fixture
def misclassifications_criterion() -> StaticNumberAvoidableMisclassificationCriterion:
    return StaticNumberAvoidableMisclassificationCriterion(
        allow_reduction=True,
        avoidable_misclassification_threshold=10,
        expected_accuracy=0.9,
    )


def test_initialization(performance_trigger_config: PerformanceTriggerConfig) -> None:
    trigger = PerformanceTrigger(performance_trigger_config)
    assert trigger._sample_left_until_detection == performance_trigger_config.evaluation_interval_data_points

    assert len(trigger.decision_policies) == 1
    assert isinstance(
        trigger.decision_policies["static_threshold"],
        StaticPerformanceThresholdDecisionPolicy,
    )

    assert not trigger._triggered_once
    assert not trigger.model_refresh_needed

    assert trigger.most_recent_model_id is None


@patch.object(PerformanceTriggerMixin, "_init_trigger")
def test_init_trigger(
    mock_init_trigger: MagicMock,
    performance_trigger_config: PerformanceTriggerConfig,
    dummy_pipeline_config: ModynPipelineConfig,
    dummy_system_config: ModynConfig,
) -> None:
    trigger_context = TriggerContext(PIPELINE_ID, dummy_pipeline_config, dummy_system_config, BASEDIR)
    trigger = PerformanceTrigger(performance_trigger_config)
    trigger.init_trigger(context=trigger_context)
    mock_init_trigger.assert_called_once_with(trigger, trigger_context)


def test_setup_decision_policies(
    performance_trigger_config: PerformanceTriggerConfig,
    static_criterion: StaticPerformanceThresholdCriterion,
    dynamic_criterion: DynamicPerformanceThresholdCriterion,
    misclassifications_criterion: StaticNumberAvoidableMisclassificationCriterion,
) -> None:
    performance_trigger_config.decision_criteria = {
        "static_threshold": static_criterion,
        "dynamic_threshold": dynamic_criterion,
        "avoidable_misclassifications": misclassifications_criterion,
    }

    policies = _setup_decision_policies(performance_trigger_config)
    assert len(policies) == 3
    assert isinstance(policies["static_threshold"], StaticPerformanceThresholdDecisionPolicy)
    assert policies["static_threshold"].config == static_criterion

    assert isinstance(policies["dynamic_threshold"], DynamicPerformanceThresholdDecisionPolicy)
    assert policies["dynamic_threshold"].config == dynamic_criterion

    assert isinstance(
        policies["avoidable_misclassifications"],
        StaticNumberAvoidableMisclassificationDecisionPolicy,
    )
    assert policies["avoidable_misclassifications"].config == misclassifications_criterion


@patch.object(PerformanceTriggerMixin, "_inform_new_model")
def test_inform_new_model(
    mock_inform_new_model: MagicMock,
    performance_trigger_config: PerformanceTriggerConfig,
) -> None:
    trigger = PerformanceTrigger(performance_trigger_config)
    data = [(i, 100 + i) for i in range(5)]
    trigger._last_detection_interval = data
    trigger.inform_new_model(42, 43, 44.0)
    mock_inform_new_model.assert_called_once_with(trigger, 42, data)


@patch.object(PerformanceTriggerMixin, "_run_evaluation", return_value=(5, 2, {"Accuracy": 0.9}))
@patch.object(DataDensityTracker, "inform_data", return_value=None)
@patch.object(
    StaticPerformanceThresholdDecisionPolicy,
    "evaluate_decision",
    side_effect=[True, False],
)
@patch.object(PerformanceDecisionPolicy, "inform_trigger")
def test_inform(
    mock_inform_trigger: MagicMock,
    mock_evaluate_decision: MagicMock,
    mock_inform_data: MagicMock,
    mock_evaluation: MagicMock,
    performance_trigger_config: PerformanceTriggerConfig,
) -> None:
    performance_trigger_config.evaluation_interval_data_points = 4
    trigger = PerformanceTrigger(performance_trigger_config)
    assert not trigger._triggered_once

    # detection interval 1: at sample (3, 103) --> forced trigger
    trigger_results = list(trigger.inform([(i, 100 + i, 0) for i in range(6)]))
    assert len(trigger_results) == 1
    assert trigger_results == [3]
    assert trigger._triggered_once
    trigger.inform_new_model(1)
    assert len(trigger._leftover_data) == 2
    assert trigger._sample_left_until_detection == 2
    assert mock_inform_data.call_count == 1
    assert mock_evaluation.call_count == 2
    assert mock_evaluate_decision.call_count == 0
    assert trigger._last_detection_interval == [(i, 100 + i) for i in range(4)]
    assert mock_inform_data.call_args_list == [call([(i, 100 + i) for i in range(4)])]
    assert mock_evaluation.call_args_list == [
        # first time within inform, second time within inform_new_model
        call(trigger, interval_data=[(i, 100 + i) for i in range(4)]),
        call(interval_data=[(i, 100 + i) for i in range(4)]),
    ]

    # reset mocks
    mock_inform_data.reset_mock()
    mock_evaluation.reset_mock()

    # inform about 5 more samples
    # detection interval 2: at sample (7, 107) --> optional trigger, 3 samples leftover
    trigger_results = list(trigger.inform([(i, 100 + i, 0) for i in range(6, 11)]))
    assert len(trigger_results) == 1
    assert trigger_results == [1]
    assert trigger._triggered_once
    trigger.inform_new_model(2)
    assert len(trigger._leftover_data) == 3
    assert trigger._sample_left_until_detection == 1
    assert mock_inform_data.call_count == 1
    assert mock_evaluation.call_count == 2
    # not an initial trigger where decision policies are skipped
    assert mock_evaluate_decision.call_count == 1
    assert trigger._last_detection_interval == [(i, 100 + i) for i in range(4, 8)]
    assert mock_inform_data.call_args_list == [call([(i, 100 + i) for i in range(4, 8)])]
    assert mock_evaluation.call_args_list == [
        # first time within inform, second time within inform_new_model
        call(trigger, interval_data=[(i, 100 + i) for i in range(4, 8)]),
        call(interval_data=[(i, 100 + i) for i in range(4, 8)]),
    ]

    # reset mocks
    mock_inform_data.reset_mock()
    mock_evaluation.reset_mock()
    mock_evaluate_decision.reset_mock()

    # inform about 1 more sample
    # detection interval 3: at sample (11, 111) --> no trigger (because of decision policy), 0 samples leftover
    trigger_results = list(trigger.inform([(i, 100 + i, 0) for i in range(11, 12)]))
    assert len(trigger_results) == 0
    assert len(trigger._leftover_data) == 0
    assert trigger._sample_left_until_detection == 4
    assert mock_inform_data.call_count == 1
    assert mock_evaluation.call_count == 1
    assert mock_evaluate_decision.call_count == 1
    assert trigger._last_detection_interval == [(i, 100 + i) for i in range(8, 12)]
    assert mock_inform_data.call_args_list == [call([(i, 100 + i) for i in range(8, 12)])]
    assert mock_evaluation.call_args_list == [call(trigger, interval_data=[(i, 100 + i) for i in range(8, 12)])]
