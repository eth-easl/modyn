from functools import cached_property

from modyn.config.schema.pipeline.sampling.downsampling_config import UncertaintyDownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import DownsamplingMode


class UncertaintyDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(
        self,
        downsampling_config: UncertaintyDownsamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
    ):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self.balance = downsampling_config.balance
        self.remote_downsampling_strategy_name = "RemoteUncertaintyDownsamplingStrategy"

    @cached_property
    def downsampling_params(self) -> dict:
        config = super().downsampling_params

        config["score_metric"] = self.downsampling_config["score_metric"]

        config["balance"] = self.balance
        if config["balance"] and self.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
            raise ValueError("Balanced sampling (balance=True) can be used only in Sample then Batch mode.")

        return config
