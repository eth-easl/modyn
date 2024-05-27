from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class LossDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)

        self.remote_downsampling_strategy_name = "RemoteLossDownsampling"
