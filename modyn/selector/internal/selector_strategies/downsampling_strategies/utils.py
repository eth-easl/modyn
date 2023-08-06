from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils.utils import instantiate_class


def instantiate_downsampler(config: dict, maximum_keys_in_memory: int) -> AbstractDownsamplingStrategy:
    downsampling_strategy_name = config["strategy"]
    downsampling_config = config

    try:
        downsampling_class = instantiate_class(
            "modyn.selector.internal.selector_strategies.downsampling_strategies",
            downsampling_strategy_name,
            downsampling_config,
            maximum_keys_in_memory,
        )
    except ValueError:
        # Try to instantiate the class even if the shor name is used
        downsampling_class = instantiate_class(
            "modyn.selector.internal.selector_strategies.downsampling_strategies",
            downsampling_strategy_name + "DownsamplingStrategy",
            downsampling_config,
            maximum_keys_in_memory,
        )
    return downsampling_class
