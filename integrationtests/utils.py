import json
import os
import pathlib
from typing import Optional

import yaml
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.selector.internal.grpc.generated.selector_pb2 import JsonString as SelectorJsonString
from modyn.selector.internal.grpc.generated.selector_pb2 import StrategyConfig

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))
CONFIG_FILE = SCRIPT_PATH.parent.parent / "modyn" / "config" / "examples" / "modyn_config.yaml"
DEFAULT_SELECTION_STRATEGY = {"name": "NewDataStrategy", "maximum_keys_in_memory": 10}
DEFAULT_MODEL_STORAGE_CONFIG = {"full_model_strategy": {"name": "PyTorchFullModel"}}


def get_modyn_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config


def get_minimal_pipeline_config(
    num_workers: int = 1,
    strategy_config: dict = DEFAULT_SELECTION_STRATEGY,
    model_storage_config: dict = DEFAULT_MODEL_STORAGE_CONFIG,
) -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "model_storage": model_storage_config,
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
    if num_workers < 0:
        raise ValueError(f"Tried to register training with {num_workers} workers.")

    if "config" in pipeline_config["model"]:
        model_config = json.dumps(pipeline_config["model"]["config"])
    else:
        model_config = "{}"

    model_storage_config = pipeline_config["model_storage"]
    full_model_strategy = ModelStorageStrategyConfig.from_config(
        get_model_strategy(model_storage_config["full_model_strategy"])
    )
    incremental_model_strategy_config: Optional[StrategyConfig] = None
    full_model_interval: Optional[int] = None
    if "incremental_model_strategy" in model_storage_config:
        incremental_strategy = model_storage_config["incremental_model_strategy"]
        incremental_model_strategy_config = get_model_strategy(incremental_strategy)
        full_model_interval = (
            incremental_strategy["full_model_interval"] if "full_model_interval" in incremental_strategy else None
        )

    incremental_model_strategy: Optional[ModelStorageStrategyConfig] = None
    if incremental_model_strategy_config is not None:
        incremental_model_strategy = ModelStorageStrategyConfig.from_config(incremental_model_strategy_config)

    with MetadataDatabaseConnection(modyn_config) as database:
        pipeline_id = database.register_pipeline(
            num_workers=num_workers,
            model_class_name=pipeline_config["model"]["id"],
            model_config=model_config,
            amp=pipeline_config["training"]["amp"] if "amp" in pipeline_config["training"] else False,
            selection_strategy=json.dumps(pipeline_config["training"]["selection_strategy"]),
            full_model_strategy=full_model_strategy,
            incremental_model_strategy=incremental_model_strategy,
            full_model_interval=full_model_interval,
        )

    return pipeline_id


def get_model_strategy(strategy_config: dict) -> StrategyConfig:
    return StrategyConfig(
        name=strategy_config["name"],
        zip=strategy_config["zip"] if "zip" in strategy_config else None,
        zip_algorithm=strategy_config["zip_algorithm"] if "zip_algorithm" in strategy_config else None,
        config=SelectorJsonString(value=json.dumps(strategy_config["config"])) if "config" in strategy_config else None,
    )
