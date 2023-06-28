from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import dynamic_module_import


def instantiate_downsampler(config: dict, maximum_keys_in_memory: int) -> AbstractDownsamplingStrategy:
    downsampling_strategy_module = dynamic_module_import(
        "modyn.selector.internal.selector_strategies.downsampling_strategies"
    )
    if "downsampling_config" not in config or "strategy" not in config["downsampling_config"]:
        downsampling_strategy_name = "EmptyDownsamplingStrategy"
        downsampling_config = {}
    else:
        downsampling_strategy_name = config["downsampling_config"]["strategy"]
        downsampling_config = config["downsampling_config"]

    # for simplicity, you can just specify the short name (without DownsamplingStrategy)
    if not hasattr(downsampling_strategy_module, downsampling_strategy_name):
        long_name = f"{downsampling_strategy_name}DownsamplingStrategy"
        if not hasattr(downsampling_strategy_module, long_name):
            raise ValueError("Requested downsampling strategy does not exist")
        downsampling_strategy_name = long_name

    downsampling_class = getattr(downsampling_strategy_module, downsampling_strategy_name)
    return downsampling_class(downsampling_config, maximum_keys_in_memory)
