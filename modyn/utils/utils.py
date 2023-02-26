import importlib
import inspect
import pathlib
import time
from types import ModuleType
from typing import Optional
import sys
import logging

import grpc
import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError

logger = logging.getLogger(__name__)

UNAVAILABLE_PKGS = []

def dynamic_module_import(name: str) -> ModuleType:
    """
    Import a module by name to enable dynamic loading of modules from config

    Args:
        name (str): name of the module to import

    Returns:
        module: the imported module
    """
    return importlib.import_module(name)


def model_available(model_type: str) -> bool:
    # this import is moved due to circular import errors caused by modyn.models
    # importing the 'package_available_and_can_be_imported' function
    import modyn.models
    available_models = list(x[0] for x in inspect.getmembers(modyn.models, inspect.isclass))
    return model_type in available_models


def trigger_available(trigger_type: str) -> bool:
    trigger_module = dynamic_module_import("modyn.backend.supervisor.internal.triggers")
    available_triggers = list(x[0] for x in inspect.getmembers(trigger_module, inspect.isclass))
    return trigger_type in available_triggers


def validate_yaml(concrete_file: dict, schema_path: pathlib.Path) -> tuple[bool, Optional[ValidationError]]:
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


def package_available_and_can_be_imported(package: str) -> bool:
    if package in UNAVAILABLE_PKGS:
        return False

    if package in sys.modules:
        # already imported
        return True

    package_spec = importlib.util.find_spec(package)
    if package_spec is None:
        UNAVAILABLE_PKGS.append(package)
        return False

    try:
        importlib.import_module(package)
        return True
    except Exception as e:
        logger.warning(f"Importing module {package} throws exception {e}")
        UNAVAILABLE_PKGS.append(package)
        return False
