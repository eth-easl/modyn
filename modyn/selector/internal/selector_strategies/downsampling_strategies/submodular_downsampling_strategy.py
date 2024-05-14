from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.submodular_function import (
    SUBMODULAR_FUNCTIONS,
)
from modyn.utils import DownsamplingMode


class SubmodularDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, maximum_keys_in_memory: int):
        super().__init__(downsampling_config, maximum_keys_in_memory)

        self.remote_downsampling_strategy_name = "RemoteSubmodularDownsamplingStrategy"

    def _build_downsampling_params(self) -> dict:
        config = super()._build_downsampling_params()

        if self.downsampling_config.get("submodular_function"):
            raise ValueError(
                f"Please specify the submodular function used to select the datapoints. "
                f"Available functions: {SUBMODULAR_FUNCTIONS}, param submodular_function"
            )
        config["submodular_function"] = self.downsampling_config["submodular_function"]

        if self.downsampling_config.get("submodular_optimizer"):
            config["submodular_optimizer"] = self.downsampling_config["submodular_optimizer"]

        config["selection_batch"] = self.downsampling_config.get("selection_batch", 64)

        config["balance"] = self.downsampling_config.get("balance", False)
        if config["balance"] and self.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
            raise ValueError("Balanced sampling (balance=True) can be used only in Sample then Batch mode.")

        return config
