import importlib
import inspect
import pathlib
import time
from types import ModuleType
from typing import Optional

import grpc
import modyn.models
import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError


def dynamic_module_import(name: str) -> ModuleType:
    """
    Import a module by name to enable dynamic loading of modules from config

    Args:
        name (str): name of the module to import

    Returns:
        module: the imported module
    """
    return importlib.import_module(name)


def model_available(model_id: str) -> bool:
    available_models = list(
        x[0] for x in inspect.getmembers(modyn.models, inspect.isclass)
    )
    return model_id in available_models


def trigger_available(trigger_id: str) -> bool:
    trigger_module = dynamic_module_import("modyn.backend.supervisor.internal.triggers")
    available_triggers = list(
        x[0] for x in inspect.getmembers(trigger_module, inspect.isclass)
    )
    return trigger_id in available_triggers


def validate_yaml(
    concrete_file: dict, schema_path: pathlib.Path
) -> tuple[bool, Optional[ValidationError]]:
    # We might want to support different permutations here of loaded/unloaded data
    # Implement as soon as required.

    assert schema_path.is_file(), f"Schema file does not exist: {schema_path}"
    with open(schema_path, "r", encoding="utf-8") as schema_file:
        schema = yaml.safe_load(schema_file)

    try:
        validate(concrete_file, schema)
    except ValidationError as error:
        return False, error

    return True, None


def current_time_millis() -> int:
    timestamp = time.time() * 1000
    return int(round(timestamp))


def grpc_connection_established(channel: grpc.Channel, timeout_sec: int = 5) -> bool:
    """Establishes a connection to a given GRPC channel. Returns the connection status.

    Args:
        channel (grpc.Channel): The GRPC to connect to.
        timeout_sec (int): The desired timeout, in seconds.

    Returns:
        bool: The connection status of the GRPC channel.
    """
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_sec)
        return True
    except grpc.FutureTimeoutError:
        return False
