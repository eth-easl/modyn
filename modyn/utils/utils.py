import errno
import hashlib
import importlib
import importlib.util
import inspect
import logging
import os
import pathlib
import random
import sys
import tempfile
import time
from enum import Enum
from inspect import isfunction
from types import ModuleType
from typing import Any, Callable, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import grpc
import numpy as np
import torch
import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError

logger = logging.getLogger(__name__)
UNAVAILABLE_PKGS = []
SECONDS_PER_UNIT = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
MAX_MESSAGE_SIZE = 1024 * 1024 * 128  # 128 MB
EMIT_MESSAGE_PERCENTAGES = [0.25, 0.5, 0.75]

EVALUATION_TRANSFORMER_FUNC_NAME = "evaluation_transformer_function"
LABEL_TRANSFORMER_FUNC_NAME = "label_transformer_function"
BYTES_PARSER_FUNC_NAME = "bytes_parser_function"

DownsamplingMode = Enum("DownsamplingMode", ["DISABLED", "BATCH_THEN_SAMPLE", "SAMPLE_THEN_BATCH"])


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
    trigger_module = dynamic_module_import("modyn.supervisor.internal.triggers")
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


def instantiate_class(class_module_name: str, class_name: str, *class_params: Any) -> Any:
    """
    Checks whether the class exists in the given module and, if so, creates the class
    Args:
        class_module_name: name of the module, ex "modyn.selector.internal.selector_strategies.presampling_strategies"
        class_name: name of the class
        *class_params: parameters for class' __init__

    Returns:
        An instance of the required class

    """
    class_module = dynamic_module_import(class_module_name)
    if not hasattr(class_module, class_name):
        raise ValueError(f"Requested class {class_name} not found in {class_module} module.")
    actual_class = getattr(class_module, class_name)
    return actual_class(*class_params)


def flatten(non_flat_list: list[list[Any]]) -> list[Any]:
    return [item for sublist in non_flat_list for item in sublist]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_directory_writable(path: pathlib.Path) -> bool:
    # We do not check for permission bits but just try
    # since that is the most reliable solution
    # See: https://stackoverflow.com/a/25868839/1625689

    try:
        testfile = tempfile.TemporaryFile(dir=path)
        testfile.close()
    except OSError as error:
        if error.errno == errno.EACCES:  # 13
            return False
        error.filename = path
        raise

    return True


def deserialize_function(serialized_function: str, func_name: str) -> Optional[Callable]:
    """
    Use this method to deserialize a particular python function given its string representation.

    Args:
        serialized_function: the function in plaintext.
        func_name: the function name to be returned.

    Returns:
        Optional[Callable]: None if serialized_function is left empty.
    """
    if serialized_function == "":
        return None
    mod_dict: dict[str, Any] = {}
    exec(serialized_function, mod_dict)  # pylint: disable=exec-used
    if func_name not in mod_dict or not isfunction(mod_dict[func_name]):
        raise ValueError(f"Invalid function is provided. Expected a function with name {func_name}.")
    return mod_dict[func_name]


def get_partition_for_worker(worker_id: int, total_workers: int, total_num_elements: int) -> tuple[int, int]:
    """
    Returns the subset of data for a specific worker.
    This method splits the range of all elements evenly among all workers. If you e.g have 13 elements and want to split
    it among 5 workers, then workers [0, 1, 2] get 3 keys whereas workers [3, 4] get two keys.

    Args:
        worker_id: the id of the worker.
        total_workers: total amount of workers.
        total_num_elements: total amount of elements to split among the workers.

    Returns:
        tuple[int, int]: the start index (offset) and the total subset size.
    """
    if worker_id < 0 or worker_id >= total_workers:
        raise ValueError(f"Asked for worker id {worker_id}, but only have {total_workers} workers!")

    subset_size = int(total_num_elements / total_workers)
    worker_subset_size = subset_size

    threshold = total_num_elements % total_workers
    if threshold > 0:
        if worker_id < threshold:
            worker_subset_size += 1
            start_index = worker_id * (subset_size + 1)
        else:
            start_index = threshold * (subset_size + 1) + (worker_id - threshold) * subset_size
    else:
        start_index = worker_id * subset_size
    if start_index >= total_num_elements:
        start_index = 0

    return start_index, worker_subset_size


def calculate_checksum(file_path: pathlib.Path, hash_func_name: str = "blake2b", chunk_num_blocks: int = 128) -> bytes:
    """
    Returns the checksum of a file.

    Args:
        file_path: the path to the file.
        hash_func_name: the name of the hash function.
        chunk_num_blocks: size of the update step.

    Returns:
        bytes: the checksum that is calculated over the file.
    """
    assert file_path.exists() and file_path.is_file()

    hash_func = hashlib.new(hash_func_name)
    with open(file_path, "rb") as file:
        while chunk := file.read(chunk_num_blocks * hash_func.block_size):
            hash_func.update(chunk)
    return hash_func.digest()


def zip_file(
    file_path: pathlib.Path, zipped_file_path: pathlib.Path, compression: int = ZIP_DEFLATED, remove_file: bool = False
) -> None:
    """
    Zips a file.

    Args:
        file_path: the path to the file that should be zipped.
        zipped_file_path: the path to the zipped file.
        compression: the compression algorithm to be used.
        remove_file: if the file should be removed after zipping.
    """
    assert file_path.exists(), "Cannot work with non-existing file"

    with ZipFile(zipped_file_path, "w", compression=compression) as zipfile:
        zipfile.write(file_path)

    if remove_file:
        os.remove(file_path)


def unzip_file(
    zipped_file_path: pathlib.Path, file_path: pathlib.Path, compression: int = ZIP_DEFLATED, remove_file: bool = False
) -> None:
    """
    Unzips a file.

    Args:
        zipped_file_path: path to the zipped file.
        file_path: path pointing to the location where the unzipped file should be stored.
        compression: the compression algorithm to be used.
        remove_file: true if we should remove the zipped file afterwards.
    """
    with ZipFile(zipped_file_path, "r", compression=compression) as zipfile:
        assert len(zipfile.namelist()) == 1

        with open(file_path, "wb") as file:
            file.write(zipfile.read(zipfile.namelist()[0]))

    if remove_file:
        os.remove(zipped_file_path)
