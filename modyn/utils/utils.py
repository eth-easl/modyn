from types import ModuleType
import modyn.models
import inspect
import importlib


def dynamic_module_import(name: str) -> ModuleType:
    """
    Import a module by name to enable dynamic loading of modules from config

    Args:
        name (str): name of the module to import

    Returns:
        module: the imported module
    """
    #components = name.split('.')
    #mod = __import__(components[0])
    # for comp in components[1:]:
    #    mod = getattr(mod, comp)
    # return mod
    return importlib.import_module(name)


def model_available(model_id: str) -> bool:
    available_models = list(x[0] for x in inspect.getmembers(modyn.models, inspect.isclass))
    return model_id in available_models
