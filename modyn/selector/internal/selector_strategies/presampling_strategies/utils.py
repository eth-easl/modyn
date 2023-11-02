from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from modyn.selector.internal.storage_backend.abstract_storage_backend import AbstractStorageBackend
from modyn.utils import instantiate_class


def instantiate_presampler(
    config: dict, modyn_config: dict, pipeline_id: int, storage_backend: AbstractStorageBackend
) -> AbstractPresamplingStrategy:
    if "presampling_config" not in config or "strategy" not in config["presampling_config"]:
        presampling_strategy = "NoPresamplingStrategy"
        presampling_config = {}
    else:
        presampling_strategy = config["presampling_config"]["strategy"]
        presampling_config = config["presampling_config"]

    try:
        presampling_class = instantiate_class(
            "modyn.selector.internal.selector_strategies.presampling_strategies",
            presampling_strategy,
            presampling_config,
            modyn_config,
            pipeline_id,
            storage_backend,
        )
    except ValueError:
        # Try to instantiate the class even if the short name is used
        presampling_class = instantiate_class(
            "modyn.selector.internal.selector_strategies.presampling_strategies",
            presampling_strategy + "PresamplingStrategy",
            presampling_config,
            modyn_config,
            pipeline_id,
            storage_backend,
        )
    return presampling_class
