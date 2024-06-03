# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import os
import pathlib
from typing import Optional
from unittest.mock import patch

from modyn.config.schema.config import ModynConfig
from modyn.config.schema.pipeline import DataDriftTriggerConfig, ModynPipelineConfig
from modyn.supervisor.internal.triggers import DataDriftTrigger
from modyn.supervisor.internal.triggers.embedding_encoder_utils import EmbeddingEncoderDownloader
from modyn.supervisor.internal.triggers.trigger import TriggerContext
from modyn.supervisor.internal.triggers.trigger_datasets import DataLoaderInfo

BASEDIR: pathlib.Path = pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir"
PIPELINE_ID = 42
SAMPLE = (10, 1, 1)


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
    tokenizer: Optional[None],
) -> None:
    pass


def test_initialization() -> None:
    trigger = DataDriftTrigger(DataDriftTriggerConfig(detection_interval_data_points=42))
    assert trigger.config.detection_interval_data_points == 42
    assert trigger.previous_trigger_id is None
    assert trigger.previous_model_id is None
    assert not trigger.model_updated
    assert not trigger.data_cache
    assert trigger.leftover_data_points == 0


@patch.object(EmbeddingEncoderDownloader, "__init__", noop_embedding_encoder_downloader_constructor_mock)
@patch.object(DataLoaderInfo, "__init__", noop_dataloader_info_constructor_mock)
def test_init_trigger(
    dummy_pipeline_config: ModynPipelineConfig,
    dummy_system_config: ModynConfig,
) -> None:
    trigger = DataDriftTrigger(DataDriftTriggerConfig())
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    with patch("os.makedirs", return_value=None):
        trigger.init_trigger(TriggerContext(PIPELINE_ID, dummy_pipeline_config, dummy_system_config, BASEDIR))
        assert trigger.context.pipeline_id == PIPELINE_ID
        assert trigger.context.pipeline_config == dummy_pipeline_config
        assert trigger.context.modyn_config == dummy_system_config
        assert trigger.context.base_dir == BASEDIR
        assert isinstance(trigger.dataloader_info, DataLoaderInfo)
        assert isinstance(trigger.encoder_downloader, EmbeddingEncoderDownloader)


def test_inform_previous_trigger_and_data_points() -> None:
    trigger = DataDriftTrigger(DataDriftTriggerConfig())
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    trigger.inform_previous_trigger_and_data_points(42, 42)
    assert trigger.previous_trigger_id == 42
    assert trigger.previous_data_points == 42


def test_inform_previous_model_id() -> None:
    trigger = DataDriftTrigger(DataDriftTriggerConfig())
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    trigger.inform_previous_model(42)
    assert trigger.previous_model_id == 42


@patch.object(DataDriftTrigger, "detect_drift", return_value=True)
def test_inform_always_drift(test_detect_drift) -> None:
    trigger = DataDriftTrigger(DataDriftTriggerConfig(detection_interval_data_points=1))
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 5

    trigger = DataDriftTrigger(DataDriftTriggerConfig(detection_interval_data_points=2))
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 2

    trigger = DataDriftTrigger(DataDriftTriggerConfig(detection_interval_data_points=5))
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 1


@patch.object(DataDriftTrigger, "detect_drift", return_value=False)
def test_inform_no_drift(test_detect_no_drift) -> None:
    trigger = DataDriftTrigger(DataDriftTriggerConfig(detection_interval_data_points=1))
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 1

    trigger = DataDriftTrigger(DataDriftTriggerConfig(detection_interval_data_points=2))
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 1

    trigger = DataDriftTrigger(DataDriftTriggerConfig(detection_interval_data_points=5))
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 1
