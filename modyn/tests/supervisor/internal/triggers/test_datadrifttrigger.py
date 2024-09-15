# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import pathlib
from unittest.mock import MagicMock, patch

from pytest import fixture

from modyn.config.schema.pipeline import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.drift.aggregation import (
    MajorityVoteDriftAggregationStrategy,
)
from modyn.config.schema.pipeline.trigger.drift.alibi_detect import (
    AlibiDetectMmdDriftMetric,
)
from modyn.config.schema.pipeline.trigger.drift.config import AmountWindowingStrategy
from modyn.config.schema.pipeline.trigger.drift.criterion import (
    DynamicQuantileThresholdCriterion,
    ThresholdDecisionCriterion,
)
from modyn.config.schema.pipeline.trigger.drift.detection_window import (
    TimeWindowingStrategy,
)
from modyn.config.schema.pipeline.trigger.simple.data_amount import (
    DataAmountTriggerConfig,
)
from modyn.supervisor.internal.triggers import DataDriftTrigger
from modyn.supervisor.internal.triggers.amounttrigger import DataAmountTrigger
from modyn.supervisor.internal.triggers.trigger import TriggerContext
from modyn.supervisor.internal.triggers.utils.datasets.dataloader_info import (
    DataLoaderInfo,
)
from modyn.supervisor.internal.triggers.utils.model.downloader import ModelDownloader

SAMPLE = (10, 1, 1)


@fixture
def drift_trigger_config() -> DataDriftTriggerConfig:
    return DataDriftTriggerConfig(
        evaluation_interval_data_points=42,
        metrics={
            "mmd": AlibiDetectMmdDriftMetric(
                decision_criterion=ThresholdDecisionCriterion(threshold=0.5),
                num_permutations=None,
                threshold=0.5,
            )
        },
        aggregation_strategy=MajorityVoteDriftAggregationStrategy(),
    )


def noop(self) -> None:
    pass


def noop_embedding_encoder_downloader_constructor_mock(
    self,
    modyn_config: dict,
    pipeline_id: int,
    base_dir: pathlib.Path,
    model_storage_address: str,
) -> None:
    pass


def noop_dataloader_info_constructor_mock(
    self,
    pipeline_id: int,
    dataset_id: str,
    num_dataloaders: int,
    batch_size: int,
    bytes_parser: str,
    transform_list: list[str],
    storage_address: str,
    selector_address: str,
    num_prefetched_partitions: int,
    parallel_prefetch_requests: int,
    shuffle: bool,
    tokenizer: None,
) -> None:
    pass


def test_initialization(drift_trigger_config: DataDriftTriggerConfig) -> None:
    trigger = DataDriftTrigger(drift_trigger_config)
    assert trigger.config.evaluation_interval_data_points == 42
    assert trigger.most_recent_model_id is None
    assert not trigger.model_refresh_needed
    assert trigger.config.windowing_strategy.id == "AmountWindowingStrategy"


@patch.object(
    ModelDownloader,
    "__init__",
    noop_embedding_encoder_downloader_constructor_mock,
)
@patch.object(DataLoaderInfo, "__init__", noop_dataloader_info_constructor_mock)
def test_init_trigger(drift_trigger_config: DataDriftTriggerConfig, dummy_trigger_context: TriggerContext) -> None:
    trigger = DataDriftTrigger(drift_trigger_config)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    with patch("os.makedirs", return_value=None):
        trigger.init_trigger(dummy_trigger_context)
        assert trigger.context == dummy_trigger_context
        assert isinstance(trigger.dataloader_info, DataLoaderInfo)
        assert isinstance(trigger.model_downloader, ModelDownloader)


def test_inform_new_model_id(drift_trigger_config: DataDriftTriggerConfig) -> None:
    trigger = DataDriftTrigger(drift_trigger_config)
    trigger.model_refresh_needed = False
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    trigger.inform_new_model(42)
    assert trigger.most_recent_model_id == 42
    assert trigger.model_refresh_needed


@patch.object(DataDriftTrigger, "_run_detection", return_value=(True, {}))
def test_inform_always_drift(test_detect_drift: MagicMock, drift_trigger_config: DataDriftTriggerConfig) -> None:
    drift_trigger_config.evaluation_interval_data_points = 1
    trigger = DataDriftTrigger(drift_trigger_config)
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_new_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 5

    drift_trigger_config.evaluation_interval_data_points = 2
    trigger = DataDriftTrigger(drift_trigger_config)
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_new_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 2

    drift_trigger_config.evaluation_interval_data_points = 5
    trigger = DataDriftTrigger(drift_trigger_config)
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_new_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 1


@patch.object(DataDriftTrigger, "_run_detection", return_value=(False, {}))
def test_inform_no_drift(test_detect_no_drift: MagicMock, drift_trigger_config: DataDriftTriggerConfig) -> None:
    drift_trigger_config.evaluation_interval_data_points = 1
    trigger = DataDriftTrigger(drift_trigger_config)
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_new_model(num_triggers)
    assert num_triggers == 1

    drift_trigger_config.evaluation_interval_data_points = 2
    trigger = DataDriftTrigger(drift_trigger_config)
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_new_model(num_triggers)
    assert num_triggers == 1

    drift_trigger_config.evaluation_interval_data_points = 5
    trigger = DataDriftTrigger(drift_trigger_config)
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_new_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 1


@patch.object(DataDriftTrigger, "_run_detection", return_value=(False, {}))
def test_update_current_window_amount_strategy(
    mock_drift_trigger: MagicMock,
    drift_trigger_config: DataDriftTriggerConfig,
) -> None:
    drift_trigger_config.evaluation_interval_data_points = 2
    drift_trigger_config.windowing_strategy = AmountWindowingStrategy(amount_cur=3, amount_ref=3)
    trigger = DataDriftTrigger(drift_trigger_config)

    # Inform with less data than the window amount
    list(trigger.inform([(1, 102, 1), (2, 103, 1)]))
    assert len(trigger.windows.current) == 2, "Current window should contain 2 data points."

    trigger.inform_new_model(42)

    # Inform with additional data points to exceed the window size
    # idx=5: doesn't fill up whole batch --> unprocessed leftover
    list(trigger.inform([(3, 103, 1), (4, 104, 1), (5, 105, 1)]))
    assert len(trigger.windows.current) == 2, "Current window should not exceed 3 data points."
    assert trigger.windows.current[0][0] == 3, "Oldest data point should be dropped."
    assert trigger._leftover_data == [(5, 105)], "Unprocessed data should be stored."


@patch.object(DataDriftTrigger, "_run_detection", return_value=(False, {}))
def test_time_windowing_strategy_update(
    mock_drift_trigger: MagicMock,
    drift_trigger_config: DataDriftTriggerConfig,
) -> None:
    drift_trigger_config.evaluation_interval_data_points = 1
    drift_trigger_config.warmup_intervals = 0
    drift_trigger_config.windowing_strategy = TimeWindowingStrategy(limit_cur="10s", limit_ref="10s")
    trigger = DataDriftTrigger(drift_trigger_config)

    # Inform with initial data points
    list(trigger.inform([(1, 100, 1), (2, 104, 1), (3, 105, 1)]))
    assert len(trigger.windows.current) == 3, "Current window should contain 3 data points."

    # Inform with additional data points outside the time window
    list(trigger.inform([(4, 111, 1), (5, 115, 1)]))
    assert len(trigger.windows.current) == 3, "Current window should contain only recent data within 10 seconds."
    # Since the window is inclusive, we have 105 in there!
    assert trigger.windows.current[0][0] == 3, "Data points outside the time window should be dropped."


@patch.object(DataDriftTrigger, "_run_detection", return_value=(False, {}))
def test_update_current_window_amount_strategy_cross_inform(
    mock_drift_trigger: MagicMock,
    drift_trigger_config: DataDriftTriggerConfig,
) -> None:
    drift_trigger_config.warmup_intervals = 0
    drift_trigger_config.windowing_strategy = AmountWindowingStrategy(amount_cur=5, amount_ref=5)
    drift_trigger_config.evaluation_interval_data_points = 3
    trigger = DataDriftTrigger(drift_trigger_config)

    trigger_indexes = list()
    for idx in trigger.inform(
        [
            (0, 100, 1),
            (1, 100, 1),
            (2, 100, 1),
            (3, 100, 1),
            (4, 100, 1),
            (5, 100, 1),
            (6, 100, 1),
        ]
    ):
        trigger_indexes.append(idx)
        trigger.inform_new_model(idx)  # use the idx as dummy model id

    assert len(trigger_indexes) == 1, "Only the first batch should trigger."
    assert trigger_indexes == [2]

    # batch 2: no trigger -> remains in current window
    # index 6 remains int the leftover data as it doesn't fill up a batch
    assert len(trigger.windows.current) == 3
    assert list(trigger.windows.current) == [(3, 100), (4, 100), (5, 100)]
    assert trigger._leftover_data == [(6, 100)]

    # fill batch 3 --> no trigger
    assert len(list(trigger.inform([(7, 100, 1), (8, 100, 1)]))) == 0
    assert len(trigger.windows.current) == 5
    assert trigger.windows.current[0][0] == 4
    assert trigger._leftover_data == []

    assert len(list(trigger.inform([(9, 100, 1)]))) == 0, "Only the first batch should trigger."
    assert len(trigger.windows.current) == 5
    assert trigger.windows.current[0][0] == 4
    assert trigger._leftover_data == [(9, 100)]


@patch.object(
    DataDriftTrigger,
    "_run_detection",
    side_effect=[(False, {})] * 5 + [(False, {}), (True, {}), (False, {})],  # first 5: warmup
)
def test_warmup_trigger(mock_drift_trigger: DataDriftTrigger) -> None:
    trigger_config = DataDriftTriggerConfig(
        evaluation_interval_data_points=5,
        metrics={
            "mmd": AlibiDetectMmdDriftMetric(
                decision_criterion=DynamicQuantileThresholdCriterion(quantile=50, window_size=3),
            )
        },
        aggregation_strategy=MajorityVoteDriftAggregationStrategy(),
        windowing_strategy=AmountWindowingStrategy(amount_cur=3, amount_ref=3),
        warmup_intervals=5,
        warmup_policy=DataAmountTriggerConfig(num_samples=7),
    )
    trigger = DataDriftTrigger(trigger_config)
    assert isinstance(trigger.warmup_trigger.trigger, DataAmountTrigger)
    assert len(trigger.windows.current) == len(trigger.windows.reference) == len(trigger.windows.current_reservoir) == 0

    # Test: We add samples from 0 to 40 in 8 batches of 5 samples each and inspect the trigger state after each batch.

    # with `evaluation_interval_data_points=5` we will detect drift every 5 samples at
    # the following indices: 5, 10, 15, 20, 25, 30, 35

    # Here are the reasons for the decisions we make at each of these points:
    # - index 4: first detection: always trigger
    # - index 9: 2nd detection: warmup trigger (warmup policy trigger at index 6)
    # - index 14: 3rd detection: warmup trigger (warmup policy trigger at index 13)
    # - index 19: 4th detection: no warmup trigger
    # - index 24: 5th detection: warmup trigger (warmup policy trigger at index 20)
    # - index 29: drift detection: run_detection --> False
    # - index 34: drift detection: run_detection --> True
    # - index 39: drift detection: run_detection --> False

    results = list(trigger.inform([(i, 100 + i, 1) for i in range(5)]))
    assert results == [4]
    assert not trigger.warmup_trigger.completed
    assert trigger.warmup_intervals[-1] == [(i, 100 + i) for i in [2, 3, 4]]  # window size 3
    assert len(trigger.windows.reference) == 3
    assert len(trigger.windows.current) == 0  # after a trigger the current window is empty

    results = list(trigger.inform([(i, 100 + i, 1) for i in range(5, 10)]))
    assert results == [4]  # index in last inform batch
    assert len(trigger.warmup_intervals) == 2
    assert not trigger.warmup_trigger.completed
    assert trigger.warmup_intervals[-1] == [(i, 100 + i) for i in [2, 3, 4]]  # from first trigger
    assert len(trigger.windows.reference) == 3
    assert len(trigger.windows.current) == 0  # after a trigger the current window is empty

    results = list(trigger.inform([(i, 100 + i, 1) for i in range(10, 15)]))
    assert results == [4]
    assert len(trigger.warmup_intervals) == 3
    assert not trigger.warmup_trigger.completed
    assert trigger.warmup_intervals[-1] == [(i, 100 + i) for i in [7, 8, 9]]
    assert len(trigger.windows.reference) == 3
    assert len(trigger.windows.current) == 0  # after a trigger the current window is empty

    results = list(trigger.inform([(i, 100 + i, 1) for i in range(15, 20)]))
    assert len(results) == 0
    assert len(trigger.warmup_intervals) == 4
    assert not trigger.warmup_trigger.completed
    assert trigger.warmup_intervals[-1] == [(i, 100 + i) for i in [12, 13, 14]]

    results = list(trigger.inform([(i, 100 + i, 1) for i in range(20, 25)]))
    assert results == [4]
    assert len(trigger.warmup_intervals) == 0
    assert trigger.warmup_trigger.completed
    assert mock_drift_trigger.call_count == 5

    results = list(trigger.inform([(i, 100 + i, 1) for i in range(25, 30)]))
    assert len(results) == 0
    assert len(trigger.warmup_intervals) == 0
    assert trigger.warmup_trigger.completed

    results = list(trigger.inform([(i, 100 + i, 1) for i in range(30, 35)]))
    assert results == [4]
    assert len(trigger.warmup_intervals) == 0
    assert trigger.warmup_trigger.completed

    results = list(trigger.inform([(i, 100 + i, 1) for i in range(35, 40)]))
    assert len(results) == 0
    assert len(trigger.warmup_intervals) == 0
    assert trigger.warmup_trigger.completed
