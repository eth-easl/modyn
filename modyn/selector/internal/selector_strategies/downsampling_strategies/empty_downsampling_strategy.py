from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class EmptyDownsamplingStrategy(AbstractDownsamplingStrategy):
    """
    This class is a little hack to just use presampling.

    """

    def __init__(self, config: dict, maximum_keys_in_memory: int) -> None:
        # just to deal with exceptions in parent class
        if "downsampling_ratio" in config:
            raise ValueError("EmptyDownsamplingStrategy has no downsampling ratio.")
        config["downsampling_ratio"] = 100

        if "sample_then_batch" in config:
            raise ValueError("EmptyDownsamplingStrategy has no downsampling mode (sample_then_batch.")
        config["sample_then_batch"] = True

        super().__init__(config, maximum_keys_in_memory)
        self.requires_remote_computation = False

    def get_downsampling_strategy(self) -> str:
        # this parameter will never be used since requires_remote_computation is False
        return ""

    def get_downsampling_params(self) -> dict:
        # this parameter will never be used since requires_remote_computation is False
        return {}

    def get_training_status_bar_scale(self) -> int:
        return 100
