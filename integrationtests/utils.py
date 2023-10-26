import json
import os
import pathlib

import yaml
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))
CONFIG_FILE = SCRIPT_PATH.parent.parent / "modyn" / "config" / "examples" / "modyn_config.yaml"
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


def init_metadata_db(modyn_config: dict) -> None:
    with MetadataDatabaseConnection(modyn_config) as database:
        database.create_tables()


def register_pipeline(pipeline_config: dict, modyn_config: dict) -> int:
    num_workers: int = pipeline_config["training"]["dataloader_workers"]
    selection_strategy: str = json.dumps(pipeline_config["training"]["selection_strategy"])

    if num_workers < 0:
        raise ValueError(f"Tried to register training with {num_workers} workers.")

    with MetadataDatabaseConnection(modyn_config) as database:
        pipeline_id = database.register_pipeline(num_workers, selection_strategy)

    return pipeline_id
