from modyn.selector.internal.selector_strategies.abstract_downsample_strategy import AbstractDownsampleStrategy


class GradNormDownsamplingStrategy(AbstractDownsampleStrategy):
    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        if "downsampled_batch_size" not in self._config:
            raise ValueError("To use GradNorm Downsampling strategy, you have to specify the downsampled_batch_size")
        self.downsampled_batch_size = self._config["downsampled_batch_size"]

        if not isinstance(self.downsampled_batch_size, int):
            raise ValueError("The downsampled batch size must be an integer")

        self._requires_remote_computation = True

    def get_downsampling_strategy(self) -> str:
        return "RemoteGradNormDownsampling"

    def get_downsampling_params(self) -> dict:
        return {"downsampled_batch_size": self.downsampled_batch_size}
