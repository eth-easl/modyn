from modyn.config.schema.pipeline import SingleDownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import instantiate_class


def instantiate_downsampler(
    downsampling_config: SingleDownsamplingConfig, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int
) -> AbstractDownsamplingStrategy:
    downsampling_strategy_name = downsampling_config.strategy

    downsampling_class = instantiate_class(
        "modyn.selector.internal.selector_strategies.downsampling_strategies",
        downsampling_strategy_name + "DownsamplingStrategy",
        downsampling_config,
        modyn_config,
        pipeline_id,
        maximum_keys_in_memory,
    )
    return downsampling_class
