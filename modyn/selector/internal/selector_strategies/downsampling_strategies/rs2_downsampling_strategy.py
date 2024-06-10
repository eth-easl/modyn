from modyn.config.schema.pipeline import RS2DownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class RS2DownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(
        self,
        downsampling_config: RS2DownsamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
    ):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self.remote_downsampling_strategy_name = "RemoteRS2Downsampling"
        self.with_replacement = downsampling_config.with_replacement

    @property
    def downsampling_params(self) -> dict:
        config = super().downsampling_params
        config["replacement"] = self.with_replacement
        return config
