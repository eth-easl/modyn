from pathlib import Path

import yaml

from .schema.client_config import ModynClientConfig


def read_client_config(path: Path) -> ModynClientConfig:
    """Read a Modyn client configuration file and validate it.

    Args:
        path: Path to the configuration file.

    Returns:
        The validated modyn client config as a pydantic model.
    """
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return ModynClientConfig.model_validate(config)
