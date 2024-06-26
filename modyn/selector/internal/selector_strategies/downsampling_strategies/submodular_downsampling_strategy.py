from functools import cached_property

from modyn.config.schema.pipeline.sampling.downsampling_config import SubmodularDownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import DownsamplingMode


class SubmodularDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(
        self,
        downsampling_config: SubmodularDownsamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
    ):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self.submodular_function = downsampling_config.submodular_function
        self.submodular_optimizer = downsampling_config.submodular_optimizer
        self.selection_batch = downsampling_config.selection_batch
        self.balance = downsampling_config.balance
        self.full_grad_approximation = downsampling_config.full_grad_approximation
        self.remote_downsampling_strategy_name = "RemoteSubmodularDownsamplingStrategy"

    @cached_property
    def downsampling_params(self) -> dict:
        config = super().downsampling_params
        config["submodular_function"] = self.submodular_function
        config["submodular_optimizer"] = self.submodular_optimizer
        config["selection_batch"] = self.selection_batch
        config["balance"] = self.balance
        config["full_grad_approximation"] = self.full_grad_approximation
        if config["balance"] and self.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
            raise ValueError("Balanced sampling (balance=True) can be used only in Sample then Batch mode.")

        return config
