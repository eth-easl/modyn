from pathlib import Path

import pytest
from modyn.config import read_modyn_config, read_pipeline
from modynclient.config import read_client_config

MODYN_PIPELINE_CONFIG_PATHS = [
    "modynclient/config/examples/dummy.yaml",
    "modynclient/config/examples/mnist.yaml",
    "benchmark/mnist/mnist.yaml",
    "benchmark/wildtime_benchmarks/example_pipelines/arxiv.yaml",
    "benchmark/wildtime_benchmarks/example_pipelines/fmow.yaml",
    "benchmark/wildtime_benchmarks/example_pipelines/huffpost.yaml",
    "benchmark/wildtime_benchmarks/example_pipelines/yearbook.yaml",
    "benchmark/wildtime_benchmarks/example_pipelines/data_drift_trigger/arxiv_datadrift.yaml",
    "benchmark/wildtime_benchmarks/example_pipelines/data_drift_trigger/huffpost_datadrift.yaml",
    "benchmark/wildtime_benchmarks/example_pipelines/data_drift_trigger/yearbook_datadrift.yaml",
]

MODYN_SYSTEM_CONFIG_PATHS = ["modyn/config/examples/modyn_config.yaml"]

MODYN_CLIENT_CONFIG_PATHS = [
    "modynclient/config/examples/modyn_client_config.yaml",
    "modynclient/config/examples/modyn_client_config_container.yaml",
]


@pytest.mark.parametrize("config_path", MODYN_PIPELINE_CONFIG_PATHS)
def test_pipeline_config_integrity(config_path: str) -> None:
    file = Path(config_path)
    assert file.exists(), f"Pipeline config file {config_path} does not exist."
    read_pipeline(file)


@pytest.mark.parametrize("config_path", MODYN_SYSTEM_CONFIG_PATHS)
def test_system_config_integrity(config_path: str) -> None:
    file = Path(config_path)
    assert file.exists(), f"System config file {config_path} does not exist."
    read_modyn_config(file)


@pytest.mark.parametrize("config_path", MODYN_CLIENT_CONFIG_PATHS)
def test_client_config_integrity(config_path: str) -> None:
    file = Path(config_path)
    assert file.exists(), f"Client config file {config_path} does not exist."
    read_client_config(file)
