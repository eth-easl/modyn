from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from modyn.utils import instantiate_class


def instantiate_presampler(config: dict, modyn_config: dict, pipeline_id: int) -> AbstractPresamplingStrategy:
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
        )
    except ValueError:
        # Try to instantiate the class even if the short name is used
        presampling_class = instantiate_class(
            "modyn.selector.internal.selector_strategies.presampling_strategies",
            presampling_strategy + "PresamplingStrategy",
            presampling_config,
            modyn_config,
            pipeline_id,
        )
    return presampling_class
