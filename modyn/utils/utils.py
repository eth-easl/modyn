import importlib
import importlib.util
import inspect
import logging
import pathlib
import sys
import time
from types import ModuleType
from typing import Any, Iterable, Optional

import grpc
import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from sqlalchemy.orm import Query

logger = logging.getLogger(__name__)
UNAVAILABLE_PKGS = []
SECONDS_PER_UNIT = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


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
    import modyn.models  # pylint: disable=import-outside-toplevel

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


def validate_timestr(timestr: str) -> bool:
    if timestr[-1] not in SECONDS_PER_UNIT:
        return False

    if not timestr[:-1].isdigit():
        return False

    return True


def convert_timestr_to_seconds(timestr: str) -> int:
    return int(timestr[:-1]) * SECONDS_PER_UNIT[timestr[-1]]


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
    except Exception as exception:  # pylint: disable=broad-except
        logger.warning(f"Importing module {package} throws exception {exception}")
        UNAVAILABLE_PKGS.append(package)
        return False


# TODO(MaxiBoether): return type?


def window_query(query: Query, column: str, windowsize: int, ordering_required: bool) -> Iterable[Any]:
    """ "Break a Query into chunks on a given column.
    Returns Iterator over chunks."""

    query = query.add_columns(column)
    if ordering_required:
        query = query.order_by(column)
    last_id = None

    while True:
        subq = query
        if last_id is not None:
            subq = subq.filter(column > last_id)
        chunk = subq.limit(windowsize).all()
        if not chunk:
            break
        last_id = chunk[-1][-1]
        yield chunk


def flatten(l):
    return [item for sublist in l for item in sublist]
