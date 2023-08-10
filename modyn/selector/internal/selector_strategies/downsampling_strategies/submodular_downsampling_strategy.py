from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.submodular_function import (
    SUBMODULAR_FUNCTIONS,
)


class SubmodularDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, maximum_keys_in_memory: int):
        super().__init__(downsampling_config, maximum_keys_in_memory)

        self.remote_downsampling_strategy_name = "RemoteSubmodularDownsamplingStrategy"

    def _build_downsampling_params(self) -> dict:
        config = super()._build_downsampling_params()

        if "submodular_function" not in self.downsampling_config:
            raise ValueError(
                f"Please specify the submodular function used to select the datapoints. "
                f"Available functions: {SUBMODULAR_FUNCTIONS}, param submodular_function"
            )
        config["submodular_function"] = self.downsampling_config["submodular_function"]

        if "submodular_optimizer" in self.downsampling_config:
            config["submodular_optimizer"] = self.downsampling_config["submodular_optimizer"]

        config["deepcore_args"] = self.downsampling_config.get("deepcore_args", {})
        return config
