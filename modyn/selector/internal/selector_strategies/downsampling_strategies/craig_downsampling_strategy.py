from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class CraigDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, maximum_keys_in_memory: int):
        super().__init__(downsampling_config, maximum_keys_in_memory)

        self.remote_downsampling_strategy_name = "RemoteCraigDownsampling"

    def _build_downsampling_params(self) -> dict:
        config = super()._build_downsampling_params()
        config["args"] = self.downsampling_config.get("args", {})
        return config
