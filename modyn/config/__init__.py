from pathlib import Path

import yaml

from .schema import *  # noqa
from .schema import ModynConfig, ModynPipelineConfig


def read_modyn_config(path: Path) -> ModynConfig:
    """Read a Modyn configuration file and validate it.

    Args:
        path: Path to the configuration file.

    Returns: e
        The validated Modyn config as a pydantic model.
    """
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return ModynConfig.model_validate(config)


def read_pipeline(path: Path) -> ModynPipelineConfig:
    """Read a pipeline configuration file.

    Args:
        path: Path to the pipeline configuration file.

    Returns:
        The validated pipeline configuration as a pydantic model.
    """
    with open(path, "r", encoding="utf-8") as pipeline_file:
        pipeline = yaml.safe_load(pipeline_file)
    return ModynPipelineConfig.model_validate(pipeline)
