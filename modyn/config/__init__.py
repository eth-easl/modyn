from pathlib import Path

import yaml
from modyn.config.schema.config import ModynConfig


def read_modyn_config(path: Path) -> ModynConfig:
    """Read a Modyn configuration file and validate it.

    Args:
        path: Path to the configuration file.

    Returns:
        The validated Modyn as a pydantic model.
    """
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return ModynConfig.model_validate(config)
