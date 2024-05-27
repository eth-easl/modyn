from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import DownsamplingMode


class GradMatchDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)

        self.remote_downsampling_strategy_name = "RemoteGradMatchDownsamplingStrategy"

    def _build_downsampling_params(self) -> dict:
        config = super()._build_downsampling_params()
        config["balance"] = self.downsampling_config.get("balance", False)
        if config["balance"] and self.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
            raise ValueError("Balanced sampling (balance=True) can be used only in Sample then Batch mode.")
        return config
