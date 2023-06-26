from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class EmptyDownsamplingStrategy(AbstractDownsamplingStrategy):
    """
    This class is a little hack to just use presampling.

    """

    def __init__(self, config: dict):
        # just to deal with exceptions in parent class
        if "downsampled_batch_size" not in config:
            config["downsampled_batch_size"] = 0

        super().__init__(config)
        self.requires_remote_computation = False

    def get_downsampling_strategy(self) -> str:
        # this parameter will never be used since requires_remote_computation is False
        return ""

    def get_downsampling_params(self) -> dict:
        # this parameter will never be used since requires_remote_computation is False
        return {}
