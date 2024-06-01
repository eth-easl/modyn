from typing import Dict

from modyn.config.schema.pipeline_component.sampling.downsampling_config import NoDownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class NoDownsamplingStrategy(AbstractDownsamplingStrategy):
    """
    This class is a little hack to just use presampling.

    """

    def __init__(
        self,
        downsampling_config: NoDownsamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
    ):
        # just to deal with exceptions in parent class
        downsampling_config.sample_then_batch = True  # useless, just to avoid ValueErrors

        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self.requires_remote_computation = False
        self.remote_downsampling_strategy_name = ""
        self.downsampling_params: Dict[None, None] = {}
