from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from modyn.utils import dynamic_module_import


def instantiate_presampler(
    config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int
) -> AbstractPresamplingStrategy:
    presampling_strategy_module = dynamic_module_import(
        "modyn.selector.internal.selector_strategies.presampling_strategies"
    )
    if "presampling_strategy" not in config:
        presampling_strategy = "EmptyPresamplingStrategy"
    else:
        presampling_strategy = config["presampling_strategy"]

    # for simplicity, you can just specify the short name (without PresamplingStrategy)
    if not hasattr(presampling_strategy_module, presampling_strategy):
        long_name = f"{presampling_strategy}PresamplingStrategy"
        if not hasattr(presampling_strategy_module, long_name):
            raise ValueError("Requested presampling strategy does not exist")
        presampling_strategy = long_name
    presampling_class = getattr(presampling_strategy_module, presampling_strategy)
    return presampling_class(
        config,
        modyn_config,
        pipeline_id,
        maximum_keys_in_memory,
    )
