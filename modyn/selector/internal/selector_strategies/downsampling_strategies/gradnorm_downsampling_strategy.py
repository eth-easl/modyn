from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class GradNormDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, maximum_keys_in_memory: int):
        super().__init__(downsampling_config, maximum_keys_in_memory)

        self.remote_downsampling_strategy_name = "RemoteGradNormDownsampling"
