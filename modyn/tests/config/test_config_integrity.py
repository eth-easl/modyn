from pathlib import Path

import pytest
from modyn.config import read_modyn_config, read_pipeline
from modynclient.config import read_client_config

"""Directories can either be file paths or directories.
If they are directories, all yaml files in the directory will be used."""

PROJECT_ROOT = Path(__file__).parents[3]

MODYN_PIPELINE_CONFIG_PATHS: list[str] = [
    "benchmark/mnist",
    "benchmark/wildtime_benchmarks",
]

MODYN_SYSTEM_CONFIG_PATHS = ["modyn/config/examples/modyn_config.yaml"]

MODYN_CLIENT_CONFIG_PATHS = [
    "modynclient/config/examples/modyn_client_config.yaml",
    "modynclient/config/examples/modyn_client_config_container.yaml",
]


def _discover_files(paths: list[str]) -> list[str]:
    files = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.is_dir():
            files.extend([str(f.absolute()) for f in p.rglob("*.yaml")])
            files.extend([str(f.absolute()) for f in p.rglob("*.yml")])
        else:
            if p.suffix == ".yaml" or p.suffix == ".yml":
                files.append(str(p.absolute()))
    return files


@pytest.mark.parametrize("config_path", _discover_files(MODYN_PIPELINE_CONFIG_PATHS))
def test_pipeline_config_integrity(config_path: str) -> None:
    file = PROJECT_ROOT / config_path
    assert file.exists(), f"Pipeline config file {config_path} does not exist."
    read_pipeline(file)


@pytest.mark.parametrize("config_path", _discover_files(MODYN_SYSTEM_CONFIG_PATHS))
def test_system_config_integrity(config_path: str) -> None:
    file = PROJECT_ROOT / config_path
    assert file.exists(), f"System config file {config_path} does not exist."
    read_modyn_config(file)


@pytest.mark.parametrize("config_path", _discover_files(MODYN_CLIENT_CONFIG_PATHS))
def test_client_config_integrity(config_path: str) -> None:
    file = PROJECT_ROOT / config_path
    assert file.exists(), f"Client config file {config_path} does not exist."
    read_client_config(file)
