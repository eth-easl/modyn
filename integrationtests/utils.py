import os
import pathlib

import yaml

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))
CONFIG_FILE = SCRIPT_PATH.parent.parent / "modyn" / "config" / "examples" / "modyn_config.yaml"


def get_modyn_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config
