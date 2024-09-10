import os
import pathlib
from unittest.mock import MagicMock, patch

from pytest import fixture

from modyn.config.schema.pipeline.evaluation.config import EvalDataConfig
from modyn.config.schema.pipeline.evaluation.metrics import (
    AccuracyMetricConfig,
    F1ScoreMetricConfig,
)
from modyn.config.schema.pipeline.trigger.performance.performance import (
    PerformanceTriggerEvaluationConfig,
    _InternalPerformanceTriggerConfig,
)
from modyn.supervisor.internal.triggers.performance.data_density_tracker import (
    DataDensityTracker,
)
from modyn.supervisor.internal.triggers.performance.performance_tracker import (
    PerformanceTracker,
)
from modyn.supervisor.internal.triggers.performance.performancetrigger_mixin import (
    PerformanceTriggerMixin,
)
from modyn.supervisor.internal.triggers.trigger import TriggerContext
from modyn.supervisor.internal.triggers.utils.model.downloader import ModelDownloader

PIPELINE_ID = 42
SAMPLE = (10, 1, 1)
BASEDIR = pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir"


@fixture
def performance_trigger_mixin_config() -> _InternalPerformanceTriggerConfig:
    return _InternalPerformanceTriggerConfig(
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
    )


def test_create_performance_trigger(
    performance_trigger_mixin_config: _InternalPerformanceTriggerConfig,
    dummy_trigger_context: TriggerContext,
) -> None:
    trigger = PerformanceTriggerMixin(performance_trigger_mixin_config)
    assert trigger.config == performance_trigger_mixin_config
    assert trigger.context is None

    assert isinstance(trigger.data_density, DataDensityTracker)
    assert trigger.data_density.batch_memory.maxlen == performance_trigger_mixin_config.data_density_window_size
    assert isinstance(trigger.performance_tracker, PerformanceTracker)

    assert not trigger.model_refresh_needed
    assert trigger.most_recent_model_id is None
    assert trigger.dataloader_info is None
    assert trigger.model_downloader is None
    assert trigger.sf_model is None

    assert len(trigger._metrics) == 2
    assert trigger._label_transformer_function is None


@patch.object(PerformanceTriggerMixin, "_init_dataloader_info", return_value=None)
@patch.object(PerformanceTriggerMixin, "_init_model_downloader", return_value=None)
def test_init_trigger(
    mock_init_model_downloader: MagicMock,
    mock_init_dataloader_info: MagicMock,
    performance_trigger_mixin_config: _InternalPerformanceTriggerConfig,
    dummy_trigger_context: TriggerContext,
) -> None:
    trigger = PerformanceTriggerMixin(performance_trigger_mixin_config)
    trigger._init_trigger(context=dummy_trigger_context)
    assert trigger.context == dummy_trigger_context
    mock_init_model_downloader.assert_called_once()
    mock_init_dataloader_info.assert_called_once()


@patch.object(PerformanceTriggerMixin, "_run_evaluation", side_effect=[(5, 2, {"Accuracy": 0.6})])
@patch.object(PerformanceTracker, "inform_trigger", return_value=None)
def test_inform_new_model(
    mock_inform_trigger: MagicMock,
    mock_evaluation: MagicMock,
    performance_trigger_mixin_config: _InternalPerformanceTriggerConfig,
) -> None:
    trigger = PerformanceTriggerMixin(performance_trigger_mixin_config)
    last_detection_interval = [(i, 100 + i) for i in range(5)]
    assert not trigger.model_refresh_needed
    trigger._inform_new_model(42, last_detection_interval)
    assert trigger.most_recent_model_id == 42
    assert trigger.model_refresh_needed  # would be reset if _run_evaluation wasn't mocked

    # distabled
    # mock_evaluation.assert_called_once_with(interval_data=last_detection_interval)
    # mock_inform_trigger.assert_called_once_with(
    #     num_samples=5, num_misclassifications=2, evaluation_scores={"Accuracy": 0.6}
    # )
    mock_evaluation.assert_not_called()


# Note: we don't test _run_evaluation as this would require more mocking than actual testing


def test_init_dataloader_info(
    performance_trigger_mixin_config: _InternalPerformanceTriggerConfig,
    dummy_trigger_context: TriggerContext,
) -> None:
    trigger = PerformanceTriggerMixin(performance_trigger_mixin_config)
    trigger.context = dummy_trigger_context

    trigger._init_dataloader_info()

    assert trigger.dataloader_info is not None


@patch.object(ModelDownloader, "connect_to_model_storage")
def test_init_model_downloader(
    mock_connect_to_model_storage: MagicMock,
    performance_trigger_mixin_config: _InternalPerformanceTriggerConfig,
    dummy_trigger_context: TriggerContext,
) -> None:
    trigger = PerformanceTriggerMixin(performance_trigger_mixin_config)
    trigger.context = dummy_trigger_context

    trigger._init_model_downloader()

    assert trigger.model_downloader is not None
