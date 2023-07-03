from typing import Dict

from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class EmptyDownsamplingStrategy(AbstractDownsamplingStrategy):
    """
    This class is a little hack to just use presampling.

    """

    def __init__(self, downsampling_config: dict, maximum_keys_in_memory: int) -> None:
        # just to deal with exceptions in parent class
        if "ratio" in downsampling_config:
            raise ValueError("EmptyDownsamplingStrategy has no downsampling ratio.")
        downsampling_config["ratio"] = 100

        if "sample_then_batch" in downsampling_config:
            raise ValueError("EmptyDownsamplingStrategy has no downsampling mode (sample_then_batch parameter).")
        downsampling_config["sample_then_batch"] = True  # useless, just to avoid ValueErrors

        super().__init__(downsampling_config, maximum_keys_in_memory)
        self.requires_remote_computation = False
        self.remote_downsampling_strategy_name = ""
        self.downsampling_params: Dict[None, None] = {}
