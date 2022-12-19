from types import ModuleType
import modyn.models
import inspect
import importlib
import yaml
import pathlib
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from typing import Optional


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
    available_models = list(x[0] for x in inspect.getmembers(modyn.models, inspect.isclass))
    return model_id in available_models


def validate_yaml(concrete_file: dict, schema_path: pathlib.Path) -> tuple[bool, Optional[ValidationError]]:
    # We might want to support different permutations here of loaded/unloaded data
    # Implement as soon as required.

    assert schema_path.is_file(), f"Schema file does not exist: {schema_path}"
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)

    try:
        validate(concrete_file, schema)
    except ValidationError as e:
        #logger.error(f"Error while validating pipeline configuration file for schema-compliance: {e.message}")
        # logger.error(e)
        return False, e

    return True, None
