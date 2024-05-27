from typing import Dict

from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class NoDownsamplingStrategy(AbstractDownsamplingStrategy):
    """
    This class is a little hack to just use presampling.

    """

    def __init__(self, downsampling_config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        # just to deal with exceptions in parent class
        if downsampling_config.get("ratio"):
            raise ValueError("NoDownsamplingStrategy has no downsampling ratio.")
        downsampling_config["ratio"] = 100

        if downsampling_config.get("sample_then_batch"):
            raise ValueError("NoDownsamplingStrategy has no downsampling mode (sample_then_batch parameter).")
        downsampling_config["sample_then_batch"] = True  # useless, just to avoid ValueErrors

        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self.requires_remote_computation = False
        self.remote_downsampling_strategy_name = ""
        self.downsampling_params: Dict[None, None] = {}
