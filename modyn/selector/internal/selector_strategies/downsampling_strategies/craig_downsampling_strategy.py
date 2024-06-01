from modyn.config.schema.pipeline.sampling.downsampling_config import CraigDownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import DownsamplingMode


class CraigDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(
        self,
        downsampling_config: CraigDownsamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
    ):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self.selection_batch = downsampling_config.selection_batch
        self.balance = downsampling_config.balance
        self.greedy = downsampling_config.greedy
        self.remote_downsampling_strategy_name = "RemoteCraigDownsamplingStrategy"

    def _build_downsampling_params(self) -> dict:
        config = super()._build_downsampling_params()
        config["selection_batch"] = self.selection_batch
        config["balance"] = self.balance
        config["greedy"] = self.greedy
        if config["balance"] and self.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
            raise ValueError("Balanced sampling (balance=True) can be used only in Sample then Batch mode.")
        return config
