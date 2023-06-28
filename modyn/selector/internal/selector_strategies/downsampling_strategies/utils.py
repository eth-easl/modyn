from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import dynamic_module_import


def instantiate_downsampler(config: dict, maximum_keys_in_memory: int) -> AbstractDownsamplingStrategy:
    downsampling_strategy_module = dynamic_module_import(
        "modyn.selector.internal.selector_strategies.downsampling_strategies"
    )
    if "downsampling_strategy" not in config:
        downsampling_strategy = "EmptyDownsamplingStrategy"
    else:
        downsampling_strategy = config["downsampling_strategy"]

    # for simplicity, you can just specify the short name (without DownsamplingStrategy)
    if not hasattr(downsampling_strategy_module, downsampling_strategy):
        long_name = f"{downsampling_strategy}DownsamplingStrategy"
        if not hasattr(downsampling_strategy_module, long_name):
            raise ValueError("Requested presampling strategy does not exist")
        downsampling_strategy = long_name

    downsampling_class = getattr(downsampling_strategy_module, downsampling_strategy)
    return downsampling_class(config, maximum_keys_in_memory)
