import os
import pathlib

import yaml
from modyn.supervisor.supervisor import Supervisor

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))
CONFIG_FILE = SCRIPT_PATH.parent.parent / "modyn" / "config" / "examples" / "modyn_config.yaml"
EVAL_DIR = pathlib.Path(".")
DEFAULT_SELECTION_STRATEGY = {"name": "NewDataStrategy", "maximum_keys_in_memory": 10}


def get_modyn_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config


def get_minimal_pipeline_config(num_workers: int = 1, strategy_config: dict = DEFAULT_SELECTION_STRATEGY) -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "training": {
            "gpus": 1,
            "device": "cpu",
            "dataloader_workers": num_workers,
            "use_previous_model": True,
            "initial_model": "random",
            "initial_pass": {"activated": False},
            "learning_rate": 0.1,
            "batch_size": 42,
            "optimizers": [
                {"name": "default1", "algorithm": "SGD", "source": "PyTorch", "param_groups": [{"module": "model"}]},
            ],
            "optimization_criterion": {"name": "CrossEntropyLoss"},
            "checkpointing": {"activated": False},
            "selection_strategy": strategy_config,
        },
        "data": {"dataset_id": "test", "bytes_parser_function": "def bytes_parser_function(x):\n\treturn x"},
        "trigger": {"id": "DataAmountTrigger", "trigger_config": {"data_points_for_trigger": 1}},
    }


def get_supervisor(pipeline_config, modyn_config) -> Supervisor:
    return Supervisor(pipeline_config, modyn_config, EVAL_DIR)
