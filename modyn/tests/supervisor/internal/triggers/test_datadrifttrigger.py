# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import os
import pathlib
from typing import Optional
from unittest.mock import patch

import pytest
from modyn.supervisor.internal.triggers import DataDriftTrigger
from modyn.supervisor.internal.triggers.model_downloader import ModelDownloader
from modyn.supervisor.internal.triggers.trigger_datasets import DataLoaderInfo

BASEDIR: pathlib.Path = pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir"
PIPELINE_ID = 42
SAMPLE = (10, 1, 1)


def get_minimal_training_config() -> dict:
    return {
        "gpus": 1,
        "device": "cpu",
        "dataloader_workers": 1,
        "use_previous_model": True,
        "initial_model": "random",
        "learning_rate": 0.1,
        "batch_size": 42,
        "optimizers": [
            {"name": "default1", "algorithm": "SGD", "source": "PyTorch", "param_groups": [{"module": "model"}]},
        ],
        "optimization_criterion": {"name": "CrossEntropyLoss"},
        "checkpointing": {"activated": False},
        "selection_strategy": {"name": "NewDataStrategy", "maximum_keys_in_memory": 10},
    }


def get_minimal_evaluation_config() -> dict:
    return {
        "device": "cpu",
        "datasets": [
            {
                "dataset_id": "MNIST_eval",
                "bytes_parser_function": "def bytes_parser_function(data: bytes) -> bytes:\n\treturn data",
                "dataloader_workers": 2,
                "batch_size": 64,
                "metrics": [{"name": "Accuracy"}],
            }
        ],
    }


def get_minimal_trigger_config() -> dict:
    return {}


def get_minimal_pipeline_config() -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "model_storage": {"full_model_strategy": {"name": "PyTorchFullModel"}},
        "training": get_minimal_training_config(),
        "data": {"dataset_id": "test", "bytes_parser_function": "def bytes_parser_function(x):\n\treturn x"},
        "trigger": {"id": "DataDriftTrigger", "trigger_config": get_minimal_trigger_config()},
    }


def get_simple_system_config() -> dict:
    return {
        "storage": {"hostname": "test", "port": 42},
        "selector": {"hostname": "test", "port": 42},
        "model_storage": {"hostname": "test", "port": 42},
    }


def noop(self) -> None:
    pass


def noop_model_downloader_constructor_mock(
    self,
    modyn_config: dict,
    pipeline_id: int,
    device: str,
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
    tokenizer: Optional[None],
) -> None:
    pass


def test_initialization() -> None:
    trigger = DataDriftTrigger({"data_points_for_detection": 42})
    assert trigger.detection_interval == 42
    assert trigger.previous_trigger_id is None
    assert trigger.previous_model_id is None
    assert not trigger.model_updated
    assert not trigger.data_cache
    assert trigger.leftover_data_points == 0


def test_init_fails_if_invalid() -> None:
    with pytest.raises(AssertionError, match="data_points_for_detection needs to be at least 1"):
        DataDriftTrigger({"data_points_for_detection": 0})

    with pytest.raises(AssertionError, match="sample_size needs to be at least 1"):
        DataDriftTrigger({"sample_size": 0})


@patch.object(ModelDownloader, "__init__", noop_model_downloader_constructor_mock)
@patch.object(DataLoaderInfo, "__init__", noop_dataloader_info_constructor_mock)
def test_init_trigger() -> None:
    trigger = DataDriftTrigger(get_minimal_trigger_config())
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    with patch("os.makedirs", return_value=None):
        pipeline_config = get_minimal_pipeline_config()
        modyn_config = get_simple_system_config()
        trigger.init_trigger(PIPELINE_ID, pipeline_config, modyn_config, BASEDIR)
        assert trigger.pipeline_id == PIPELINE_ID
        assert trigger.pipeline_config == pipeline_config
        assert trigger.modyn_config == modyn_config
        assert trigger.base_dir == BASEDIR
        assert isinstance(trigger.dataloader_info, DataLoaderInfo)
        assert isinstance(trigger.model_downloader, ModelDownloader)


def test_inform_previous_trigger_and_data_points() -> None:
    trigger = DataDriftTrigger(get_minimal_trigger_config())
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    trigger.inform_previous_trigger_and_data_points(42, 42)
    assert trigger.previous_trigger_id == 42
    assert trigger.previous_data_points == 42


def test_inform_previous_model_id() -> None:
    trigger = DataDriftTrigger(get_minimal_trigger_config())
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    trigger.inform_previous_model(42)
    assert trigger.previous_model_id == 42


@patch.object(DataDriftTrigger, "detect_drift", return_value=True)
def test_inform_always_drift(test_detect_drift) -> None:
    trigger = DataDriftTrigger({"data_points_for_detection": 1})
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 5

    trigger = DataDriftTrigger({"data_points_for_detection": 2})
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 2

    trigger = DataDriftTrigger({"data_points_for_detection": 5})
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 1


@patch.object(DataDriftTrigger, "detect_drift", return_value=False)
def test_inform_no_drift(test_detect_no_drift) -> None:
    trigger = DataDriftTrigger({"data_points_for_detection": 1})
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 1

    trigger = DataDriftTrigger({"data_points_for_detection": 2})
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 1

    trigger = DataDriftTrigger({"data_points_for_detection": 5})
    num_triggers = 0
    for _ in trigger.inform([SAMPLE, SAMPLE, SAMPLE, SAMPLE, SAMPLE]):
        num_triggers += 1
        trigger.inform_previous_trigger_and_data_points(num_triggers, 42)
        trigger.inform_previous_model(num_triggers)
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert num_triggers == 1
