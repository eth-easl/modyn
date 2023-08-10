from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class KcenterGreedyDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, maximum_keys_in_memory: int):
        super().__init__(downsampling_config, maximum_keys_in_memory)

        self.remote_downsampling_strategy_name = "RemoteKcenterGreedyDownsamplingStrategy"

    def _build_downsampling_params(self) -> dict:
        config = super()._build_downsampling_params()
        config["deepcore_args"] = self.downsampling_config.get("deepcore_args", {})
        return config
